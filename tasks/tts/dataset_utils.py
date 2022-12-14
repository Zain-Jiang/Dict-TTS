from utils.cwt import get_lf0_cwt
import torch.optim
import torch.utils.data
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
import numpy as np
from tasks.base_task import BaseDataset
import torch
import torch.optim
import torch.utils.data
import utils
import torch.distributions
from utils.hparams import hparams


class BaseTTSDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, test_items=None, test_sizes=None, data_dir=None):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        self.ext_mel2ph = None

        def load_size():
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')

        if prefix == 'test' or hparams['infer']:
            if test_items is not None:
                self.indexed_ds, self.sizes = test_items, test_sizes
            else:
                load_size()
            if hparams['num_test_samples'] > 0:
                self.avail_idxs = [x for x in range(hparams['num_test_samples']) \
                                   if x < len(self.sizes)]
                if len(hparams['test_ids']) > 0:
                    self.avail_idxs = hparams['test_ids'] + self.avail_idxs
            else:
                self.avail_idxs = list(range(len(self.sizes)))
        else:
            load_size()
            self.avail_idxs = list(range(len(self.sizes)))

        if hparams['min_frames'] > 0:
            self.avail_idxs = [
                x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
        self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item['mel']) == self.sizes[index], (len(item['mel']), self.sizes[index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
        }

        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        return batch


class FastSpeechDataset(BaseTTSDataset):
    def __init__(self, prefix, shuffle=False, test_items=None, test_sizes=None, data_dir=None):
        super().__init__(prefix, shuffle, test_items, test_sizes, data_dir)
        self.f0_mean, self.f0_std = hparams.get('f0_mean', None), hparams.get('f0_std', None)
        self.pitch_type = hparams.get('pitch_type')
        if self.pitch_type == 'cwt':
            _, hparams['cwt_scales'] = get_lf0_cwt(np.ones(10))

    def __getitem__(self, index):
        sample = super(FastSpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        max_frames = hparams['max_frames']
        spec = sample['mel']
        T = spec.shape[0]
        phone = sample['txt_token']
        sample['energy'] = (spec.exp() ** 2).sum(-1).sqrt()
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T] if 'mel2ph' in item else None
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            if hparams.get('normalize_pitch', False):
                f0 = item["f0"]
                if len(f0 > 0) > 0 and f0[f0 > 0].std() > 0:
                    f0[f0 > 0] = (f0[f0 > 0] - f0[f0 > 0].mean()) / f0[f0 > 0].std() * hparams['f0_std'] + \
                                 hparams['f0_mean']
                    f0[f0 > 0] = f0[f0 > 0].clip(min=60, max=500)
                pitch = f0_to_coarse(f0)
                pitch = torch.LongTensor(pitch[:max_frames])
            else:
                pitch = torch.LongTensor(item.get("pitch"))[:max_frames] if "pitch" in item else None
            f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if self.pitch_type == 'cwt':
                cwt_spec = torch.Tensor(item['cwt_spec'])[:max_frames]
                f0_mean = item.get('f0_mean', item.get('cwt_mean'))
                f0_std = item.get('f0_std', item.get('cwt_std'))
                sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
            elif self.pitch_type == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None, hparams)
                f0_phlevel_sum = torch.zeros_like(phone).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(phone).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph, hparams)
        else:
            f0 = uv = torch.zeros_like(mel2ph)
            pitch = None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        batch = super(FastSpeechDataset, self).collater(samples)
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples]) if samples[0]['pitch'] is not None else None
        uv = utils.collate_1d([s['uv'] for s in samples])
        energy = utils.collate_1d([s['energy'] for s in samples], 0.0)
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        batch.update({
            'mel2ph': mel2ph,
            'energy': energy,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        })
        if self.pitch_type == 'cwt':
            cwt_spec = utils.collate_2d([s['cwt_spec'] for s in samples])
            f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
            f0_std = torch.Tensor([s['f0_std'] for s in samples])
            batch.update({'cwt_spec': cwt_spec, 'f0_mean': f0_mean, 'f0_std': f0_std})
        return batch


class FastSpeechWordDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super(FastSpeechWordDataset, self).__getitem__(index)
        item = self._get_item(index)
        max_frames = sample['mel'].shape[0]
        sample['words'] = item['words']
        sample["ph_words"] = item["ph_words"]
        sample["word_tokens"] = torch.LongTensor(item["word_tokens"])
        sample["mel2word"] = torch.LongTensor(item.get("mel2word"))[:max_frames]
        sample["ph2word"] = torch.LongTensor(item['ph2word'][:hparams['max_input_tokens']])
        return sample

    def collater(self, samples):
        batch = super(FastSpeechWordDataset, self).collater(samples)
        ph_words = [s['ph_words'] for s in samples]
        batch['ph_words'] = ph_words
        word_tokens = utils.collate_1d([s['word_tokens'] for s in samples], 0)
        batch['word_tokens'] = word_tokens
        mel2word = utils.collate_1d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        ph2word = utils.collate_1d([s['ph2word'] for s in samples], 0)
        batch['ph2word'] = ph2word
        batch['words'] = [s['words'] for s in samples]
        batch['word_lengths'] = torch.LongTensor([len(s['word_tokens']) for s in samples])
        if self.hparams['use_word_input']:
            batch['txt_tokens'] = batch['word_tokens']
            batch['txt_lengths'] = torch.LongTensor([s['word_tokens'].numel() for s in samples])
            batch['mel2ph'] = batch['mel2word']
        return batch


class DictTTSDataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, test_items=None, test_sizes=None, data_dir=None):
        super().__init__(prefix, shuffle, test_items, test_sizes, data_dir)
        import json
        import pickle
        from utils.text_encoder import TokenTextEncoder
        self.f0_mean, self.f0_std = hparams.get('f0_mean', None), hparams.get('f0_std', None)
        self.pitch_type = hparams.get('pitch_type')
        if self.pitch_type == 'cwt':
            _, hparams['cwt_scales'] = get_lf0_cwt(np.ones(10))
        word_set = json.load(open(hparams['binary_data_dir']+'/word_set.json', 'r'))
        self.token_text_encoder = TokenTextEncoder(None, vocab_list=word_set, replace_oov='<UNK>')

        # Load the pinyin encoder
        with open(hparams['binary_data_dir']+"/pinyin_encoder.pkl", 'rb') as f:
            self.pinyin_encoder = pickle.load(f)  
        # define the dict embed dataset for main thread
        self.dict_ds = None
    
    def __getitem__(self, index):
        sample = super(DictTTSDataset, self).__getitem__(index)
        item = self._get_item(index)
        max_frames = sample['mel'].shape[0]
        sample['words'] = item['words']
        sample["ph_words"] = item["ph_words"]
        sample["word_tokens"] = torch.LongTensor(item["word_tokens"])

        sample["mel2word"] = torch.LongTensor(item.get("mel2word"))[:max_frames]
        sample["ph2word"] = torch.LongTensor(item['ph2word'])
        if 'pron_modified' in item:
            sample["pron_modified"] = torch.LongTensor(item['pron_modified'])

        if self.hparams['use_dict']:
            self.get_dict_embeddings(sample)

        return sample


    def collater(self, samples):
        batch = super(DictTTSDataset, self).collater(samples)
        ph_words = [s['ph_words'] for s in samples]
        batch['ph_words'] = ph_words
        word_tokens = utils.collate_1d([s['word_tokens'] for s in samples], 0)
        batch['word_tokens'] = word_tokens
        mel2word = utils.collate_1d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        ph2word = utils.collate_1d([s['ph2word'] for s in samples], 0)
        batch['ph2word'] = ph2word
        batch['words'] = [s['words'] for s in samples]
        batch['word_lengths'] = torch.LongTensor([len(s['word_tokens']) for s in samples])
        if self.hparams['use_word_input']:
            # batch['txt_tokens'] = batch['word_tokens']
            batch['txt_lengths'] = torch.LongTensor([s['word_tokens'].numel() for s in samples])
            batch['mel2ph'] = batch['mel2word']
            if 'pron_modified' in samples[0]:
                batch["pron_modified"] = utils.collate_1d([s['pron_modified'] for s in samples], 0)
            else:
                batch["pron_modified"] = None

        if self.hparams['use_dict']:
            batch['keys'] = utils.collate_3d([s['keys'] for s in samples], pad_idx=0)
            batch['keys'] = torch.nn.functional.pad(batch['keys'], (0,0,0,0,1,1), mode='constant', value=0)
            batch['key_map'] = utils.collate_3d([s['key_map'].unsqueeze(-1) for s in samples], pad_idx=0).squeeze(-1)
            batch['key_map'] = torch.nn.functional.pad(batch['key_map'], (0,0,1,1), mode='constant', value=1)
            batch['values'] = utils.collate_3d([s['values'] for s in samples], pad_idx=0)
            batch['values'] = torch.nn.functional.pad(batch['values'], (0,0,0,0,1,1), mode='constant', value=0)
            batch['tokens_gloss'] = [s['tokens_gloss'] for s in samples]
            batch['pinyin'] = utils.collate_3d([s['pinyin'].unsqueeze(-1) for s in samples], pad_idx=0).squeeze(-1)
            batch['pinyin'] = torch.nn.functional.pad(batch['pinyin'], (0,0,1,1), mode='constant', value=0)
            batch['pinyin_map'] = utils.collate_3d([s['pinyin_map'].unsqueeze(-1) for s in samples], pad_idx=0).squeeze(-1)
            batch['pinyin_map'] = torch.nn.functional.pad(batch['pinyin_map'], (0,0,1,1), mode='constant', value=1)
        else:
            batch['keys'] = None
            batch['values'] = None
            batch['tokens_gloss'] = None

        return batch


    def get_dict_embeddings(self, sample):
        # Load the dict embed dataset for each thread
        if self.dict_ds is None:
            self.dict_ds = IndexedDataset(f'{self.data_dir}/dict_embed')

        keys, key_map, values, tokens_gloss, pinyin, pinyin_map = [], [], [], [], [], []
        
        for i, word in enumerate(sample['words'][1:-1]):
            if word in self.token_text_encoder._token_to_id:
                word_idx = self.token_text_encoder._token_to_id[word]
            else:
                word_idx = 2 # 2 for <UNK>
            embed = self.dict_ds[word_idx]
            keys.append(embed['key'])
            key_map.append(torch.Tensor(embed['key_map']))
            values.append(embed['value'])
            tokens_gloss.append(embed['tokens_gloss'])
            pinyin.append(torch.Tensor([self.pinyin_encoder.index(item) for item in embed['pinyin']]).long())
            pinyin_map.append(torch.LongTensor(embed['pinyin_map']))

        sample["keys"] = utils.collate_2d(keys, pad_idx=0)
        sample["key_map"] = utils.collate_1d(key_map, pad_idx=0)
        sample["values"] = utils.collate_2d(values, pad_idx=0)
        sample["tokens_gloss"] = tokens_gloss
        sample["pinyin"] = utils.collate_1d(pinyin, pad_idx=0)
        sample["pinyin_map"] = utils.collate_1d(pinyin_map, pad_idx=0)