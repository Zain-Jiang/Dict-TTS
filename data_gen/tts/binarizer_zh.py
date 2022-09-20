import json
import os
import pickle
from re import T
import re
import tqdm
from collections import Counter

os.environ["OMP_NUM_THREADS"] = "1"

from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.tts.data_gen_utils import get_mel2ph
from data_gen.tts.sandhi_processor import SandhiProcessor
from resemblyzer import VoiceEncoder
from utils.multiprocess_utils import chunked_multiprocess_run
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.hparams import set_hparams, hparams
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pypinyin.contrib.tone_convert import to_initials, to_finals_tone, to_finals_tone3
from utils.text_norm import NSWNormalizer

table = {ord(f): ord(t) for f, t in zip(
    u'【】（）％＃＠＆１２３４５６７８９０',
    u'[]()%#@&1234567890')}
PUNCS = '、：，。！？；'
PUNCS_2 = '!,.?;:'


class ZhBinarizer(BaseBinarizer):
    sandhi_processor = SandhiProcessor()

    def _word_encoder(self):
        fn = f"{hparams['binary_data_dir']}/word_set.json"
        if hparams['finetune_from_asr']:
            self.binarization_args['reset_word_dict'] = False
            word_set_ = json.load(open(f"data/zh-dict.json", 'r'))
            word_set = []
            for word in word_set_.keys():
                word_set += list(word)
            word_set.sort()
            json.dump(word_set, open(fn, 'w'))

        if self.binarization_args['reset_word_dict']:
            word_set = []
            for word_sent in self.item2txt.values():
                word_set += list(word_sent)
            word_set = Counter(word_set)
            total_words = sum(word_set.values())
            word_set = word_set.most_common(hparams['word_size'])
            num_unk_words = total_words - sum([x[1] for x in word_set])
            word_set = [x[0] for x in word_set]
            json.dump(word_set, open(fn, 'w'))
            print(f"| #total words: {total_words}, #unk_words: {num_unk_words}")
        else:
            word_set = json.load(open(fn, 'r'))
        print("| Word dict size: ", len(word_set), word_set[:10])

        from utils.text_encoder import TokenTextEncoder
        token_text_encoder = TokenTextEncoder(None, vocab_list=word_set, replace_oov='<UNK>')
        self.init_language_model()
        self.gen_dict_embeddings(token_text_encoder, 'data/zh-dict.json')

        return token_text_encoder

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        ph_lengths = []
        mel_lengths = []
        f0s = []
        total_sec = 0
        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()

        meta_data = list(self.meta_data(prefix))
        for m in meta_data:
            args.append(list(m) + [(self.phone_encoder, self.word_encoder), self.binarization_args])
        num_workers = self.num_workers
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
            if item is None:
                continue
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                if self.binarization_args['with_spk_embed'] else None
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            
            item['pron_modified'] = self.sandhi_processor.process_sandhi("".join([i for i in item['words'][1:-1]]), item['words'])

            builder.add_item(item)
            mel_lengths.append(item['len'])
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            total_sec += item['sec']
            if item.get('f0') is not None:
                f0s.append(item['f0'])
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @staticmethod
    def get_align(tg_fn, res):
        ph = res['ph']
        mel = res['mel']
        phone = res['phone']
        if tg_fn is not None and os.path.exists(tg_fn):
            _, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        else:
            raise BinarizationError(f"Align not found")
        ph_list = ph.split(" ")
        assert len(dur) == len(ph_list)
        mel2ph = []
        for i in range(len(dur)):
            mel2ph += [i + 1] * dur[i]
        mel2ph = np.array(mel2ph)
        if mel2ph.max() - 1 >= len(phone):
            raise BinarizationError(f"| Align does not match: {(mel2ph.max() - 1, len(phone))}")
        res['mel2ph'] = mel2ph
        res['dur'] = dur

        # char-level pitch
        if 'f0' in res:
            res['f0_ph'] = np.array([0 for _ in res['f0']], dtype=float)
            char_start_idx = 0
            f0s_char = []
            # ph_list = 0
            for idx, (f0_, ph_idx) in enumerate(zip(res['f0'], res['mel2ph'])):
                is_pinyin = ph_list[ph_idx - 1][0].isalpha()
                if not is_pinyin or ph_idx - res['mel2ph'][idx - 1] > 1:
                    if len(f0s_char) > 0:
                        res['f0_ph'][char_start_idx:idx] = sum(f0s_char) / len(f0s_char)
                    f0s_char = []
                    char_start_idx = idx
                    if not is_pinyin:
                        char_start_idx += 1
                if f0_ > 0:
                    f0s_char.append(f0_)

    @staticmethod
    def get_word(res, word_encoder):
        ph_split = res['ph'].split(" ")
        # ph side mapping to word
        ph_words = []  # ['<BOS>', 'N_AW1_', ',', 'AE1_Z_|', 'AO1_L_|', 'B_UH1_K_S_|', 'N_AA1_T_|', ....]
        ph2word = np.zeros([len(ph_split)], dtype=int)
        last_ph_idx_for_word = []  # [2, 11, ...]
        for i, ph in enumerate(ph_split):
            if ph in ['|', '#']:
                last_ph_idx_for_word.append(i)
            elif not ph[0].isalnum():
                if ph not in ['<BOS>']:
                    last_ph_idx_for_word.append(i - 1)
                last_ph_idx_for_word.append(i)
        start_ph_idx_for_word = [0] + [i + 1 for i in last_ph_idx_for_word[:-1]]
        for i, (s_w, e_w) in enumerate(zip(start_ph_idx_for_word, last_ph_idx_for_word)):
            ph_words.append(ph_split[s_w:e_w + 1])
            ph2word[s_w:e_w + 1] = i
        ph2word = ph2word.tolist()
        ph_words = ["_".join(w) for w in ph_words]

        # mel side mapping to word
        mel2word = []
        dur_word = [0 for _ in range(len(ph_words))]
        for i, m2p in enumerate(res['mel2ph']):
            word_idx = ph2word[m2p - 1]
            mel2word.append(ph2word[m2p - 1])
            dur_word[word_idx] += 1
        ph2word = [x + 1 for x in ph2word]  # 0预留给padding
        mel2word = [x + 1 for x in mel2word]  # 0预留给padding
        res['ph_words'] = ph_words  # [T_word]
        res['ph2word'] = ph2word  # [T_ph]
        res['mel2word'] = mel2word  # [T_mel]
        res['dur_word'] = dur_word  # [T_word]

        words = [x for x in res['txt']]
        if words[-1] in PUNCS_2:
            words = words[:-1]

        words = ['<BOS>'] + words + ['<EOS>']
        word_tokens = word_encoder.encode(" ".join(words))
        res['words'] = words
        res['word_tokens'] = word_tokens
        assert len([i for i in words if i != '#']) == len(ph_words), [words, ph_words]

    def preprocess_text(self, text):
        text = text.translate(table)
        text = NSWNormalizer(text).normalize(remove_punc=False).lower()
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r"", text)
        text = re.sub(rf"[A-Za-z]+", r"", text)
        text = re.sub("\(\d+\)", " ",text)
        text = re.sub("〔", " ",text)
        text = re.sub("〕", " ",text)
        return text

    def init_language_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("pretrained/roformer-chinese-base")
        self.language_model = AutoModelForMaskedLM.from_pretrained("pretrained/roformer-chinese-base")
        self.language_model.cuda()

    def get_encodings(self, text, max_token=30):
        if max_token:
            text = self.preprocess_text(text)
            text = text[:max_token]

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt") 
            for item in inputs:
                inputs[item] = inputs[item].cuda() 
            outputs = self.language_model(**inputs, output_hidden_states=True, return_dict=True)['hidden_states']
            
            shallow_feat = self.language_model.get_input_embeddings()(inputs['input_ids']).cpu().squeeze(0)

            feat = torch.zeros(9, shallow_feat.size(0), shallow_feat.size(1))
            feat[0] = shallow_feat
            for idx, item in enumerate(outputs[0:8]):
                feat[idx+1] += item.cpu().squeeze(0)
            feat = feat.mean(dim=0)
            return {'key': feat,
                    'value': feat,
                    'tokens': (['<sos>'] + self.tokenizer.tokenize(text) + ['<eos>'])}
        

    def gen_dict_embeddings(self, token_text_encoder, dict_path):
        pinyin_encoder = ['<UNK>'] # Init pinyin encoder
        
        print("Extracting embeddings from the zh-dict.")
        data_dir = hparams['binary_data_dir']
        zh_dict = json.load(open(dict_path, 'r'))

        dict_builder = IndexedDatasetBuilder(f'{data_dir}/dict_embed')
        for word in tqdm(token_text_encoder._token_to_id): # Get the word embedding from dict
            res_pinyin = None
            res_gloss = None
            gloss_length_mask = []
            pinyin_length_mask = []

            if word not in zh_dict:
                embed = {'tokens_gloss': ['O'],
                         'key': torch.zeros([3, 768]),
                         'key_map': [0,1,0],
                         'value': torch.zeros([3, 768]),
                         'pinyin': ['<UNK>'],
                         'pinyin_map': [1]
                        }
                dict_builder.add_item(embed)
                continue

            glosses = zh_dict[word]
            for pinyin in glosses:
                multi_prouncation = False
                gloss = ("".join(glosses[pinyin])).replace('～', word)
                
                if res_gloss == None:
                    res_pinyin = [to_initials(pinyin, strict=False), to_finals_tone3(pinyin, strict=False)]
                    res_gloss = self.get_encodings(gloss)
                    gloss_length_mask += [res_gloss['key'].shape[0]]
                    pinyin_length_mask += [len(res_pinyin)]

                else:
                    multi_prouncation = True
                    pinyin = [to_initials(pinyin, strict=False), to_finals_tone3(pinyin, strict=False)]
                    res_pinyin = res_pinyin + pinyin
                    res_gloss_tmp = self.get_encodings(gloss)
                    res_gloss['tokens'] += res_gloss_tmp['tokens']
                    res_gloss['key'] = torch.cat([res_gloss['key'],res_gloss_tmp['key']], dim=0)
                    res_gloss['value'] = torch.cat([res_gloss['value'],res_gloss_tmp['value']], dim=0)

                    gloss_length_mask += [res_gloss_tmp['key'].shape[0]]
                    pinyin_length_mask += [len(pinyin)]

            # Add shengmu and yunmu unit to pinyin encoder for dict construction
            for item in res_pinyin:
                if item not in pinyin_encoder:
                    pinyin_encoder.append(item)
            
            # Get key_map [word_num, max_key_length]
            key_map = []
            for idx, j in enumerate(gloss_length_mask):
                key_map += [0]
                key_map += [idx+1]*(j-2)
                key_map += [0]

            # Get value_map [word_num, max_key_length]
            pinyin_map = []
            for idx, j in enumerate(pinyin_length_mask):
                pinyin_map += [idx+1]*j
            
            embed = {'tokens_gloss': res_gloss['tokens'],
                     'key': res_gloss['key'],
                     'key_map': key_map,
                     'value': res_gloss['value'],
                     'pinyin': res_pinyin,
                     'pinyin_map': pinyin_map
                    }

            dict_builder.add_item(embed)

        dict_builder.finalize()
        print(len(pinyin_encoder))
        pickle.dump(pinyin_encoder, open(f"{data_dir}/pinyin_encoder.pkl", 'wb'))
        print("Extracting process finished.")


if __name__ == "__main__":
    set_hparams()
    ZhBinarizer().process()



"""
    glosses: 一个词的多个释义
    get_encoding(glosses): 
        input: self.language_model.get_input_embeddings()(inputs['input_ids'])
        output：经过xlm之后的embedding输出
        tokens：sos + tokenize之后的text + eos
    
    key：xlm之后的embedding
    更新mask

    最后的输出：
        key：对ph做的结果
        value：整个拼音字母表

    先对拼音做，然后对word做
"""