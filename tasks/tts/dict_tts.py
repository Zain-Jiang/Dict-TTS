import os
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils
from modules.fastspeech.multi_window_disc import Discriminator
from modules.dict_tts.model import PortaSpeech_dict
from tasks.tts.ps_adv import PortaSpeechAdvTask
from tasks.tts.dataset_utils import DictTTSDataset
from utils.hparams import hparams
from utils.plot import spec_to_figure, attn_to_figure_unmerged
from data_gen.tts.data_gen_utils import get_pitch
from vocoders.base_vocoder import get_vocoder_cls


class DictTTSTask(PortaSpeechAdvTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = DictTTSDataset

    def build_model(self):
        self.model = PortaSpeech_dict(self.phone_encoder)
        utils.print_arch(self.model, 'Generator')
        self.gen_params = self.model.parameters()
        self.build_disc_model()

        return self.model

    def build_disc_model(self):
        self.disc_params = []
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3),
            norm_type=hparams['disc_norm'], reduction=hparams['disc_reduction']
        )
        self.disc_params += list(self.mel_disc.parameters())
        utils.print_arch(self.mel_disc, model_name='Mel Disc')

    def run_model(self, model, sample, return_output=False, training_post_glow=False):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = self.model((sample['word_tokens'], sample['txt_tokens']),
                            sample['pron_modified'],
                            (None, None, None),
                            ph2word=sample['ph2word'],
                            mel2word=sample['mel2word'],
                            mel2ph=sample['mel2ph'],
                            word_len=sample['word_lengths'].max(),
                            dict_msg=(sample["keys"], sample["values"], sample["key_map"], sample["pinyin"], sample["pinyin_map"]),
                            tgt_mels=sample['mels'],
                            spk_embed=spk_embed,
                            infer=False,
                            forward_post_glow=training_post_glow,
                            two_stage=hparams['two_stage'])
        losses = {}
        if (training_post_glow or not hparams['two_stage']) and hparams['use_post_glow']:
            losses['postflow'] = output['postflow']
        if not training_post_glow or not hparams['two_stage'] or not self.training:
            losses['kl'] = output['kl'] * hparams['lambda_kl']
            self.add_mel_loss(output['mel_out'], sample['mels'], losses)
            if hparams['dur_level'] == 'word':
                self.add_dur_loss(
                    output['dur'], sample['mel2word'], sample['word_lengths'], sample['txt_tokens'], losses)
                if 'attn' in output:
                    self.get_attn_stats(output['attn'], sample, losses)
            else:
                self.add_dur_loss(
                    output['dur'], sample['mel2ph'], sample['txt_lengths'], sample['txt_tokens'], losses)
       
        losses['kl'] = torch.clamp(losses['kl'], 0.002, None)
        
        if not return_output:
            return losses
        else:
            return losses, output

    def _training_step(self, sample, batch_idx, optimizer_idx):
        log_outputs = {}
        loss_weights = {}
        disc_start = self.global_step > hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            log_outputs, model_out = self.run_model(self.model, sample, return_output=True)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    log_outputs['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    log_outputs['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    log_outputs["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    log_outputs["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    log_outputs["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    log_outputs["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(log_outputs) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])
        return total_loss, log_outputs

    def validation_step(self, sample, batch_idx):
        training_post_glow = self.global_step >= hparams['post_glow_training_start'] \
                             and hparams['use_post_glow']
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(
            self.model, sample, return_output=True, training_post_glow=training_post_glow)
        outputs['nsamples'] = sample['nsamples']
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
            model_out = self.model((sample['word_tokens'], sample['txt_tokens']),
                                   sample['pron_modified'],
                                   (None, None, None),
                                   ph2word=sample['ph2word'],
                                   mel2word=sample['mel2word'],
                                   word_len=sample['word_lengths'].max(),
                                   dict_msg=(sample["keys"], sample["values"], sample["key_map"], sample["pinyin"], sample["pinyin_map"]),
                                   spk_embed=spk_embed,
                                   infer=True,
                                   forward_post_glow=training_post_glow,
                                   two_stage=hparams['two_stage'])
            vmin = hparams['mel_vmin']
            vmax = hparams['mel_vmax']
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu())
            if self.global_step > 0:
                self.logger.add_audio(f'wav_{batch_idx}', wav_pred, self.global_step, hparams["audio_sample_rate"])
                self.logger.add_figure(
                    f'mel_{batch_idx}', spec_to_figure(torch.cat([sample['mels'][0], model_out['mel_out'][0]],dim=1), vmin, vmax), self.global_step)
            wav_pred = self.vocoder.spec2wav(model_out['mel_out_fvae'][0].cpu())
            if self.global_step > 0:
                self.logger.add_audio(f'wav_fvae_{batch_idx}', wav_pred, self.global_step, hparams["audio_sample_rate"])
                self.logger.add_figure(
                    f'mel_fvae_{batch_idx}', spec_to_figure(model_out['mel_out_fvae'][0], vmin, vmax), self.global_step)
            if hparams['dur_level'] == 'word':
                if 'attn' in model_out:
                    self.logger.add_figure(
                        f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)
                    self.plot_dur(batch_idx, sample, model_out)
            else:
                self.plot_dur_ph(batch_idx, sample, model_out)

            if 'dict_attn' in model_out and model_out['dict_attn'] != None:
                self.logger.add_figure(
                    f'dict_attn_layer1_unmerged_{batch_idx}', attn_to_figure_unmerged(model_out['dict_attn'][0][0], sample, vmin=0, vmax=1), self.global_step)

        return outputs

    def test_step(self, sample, batch_idx):
        forward_post_glow = self.global_step >= hparams['post_glow_training_start'] + 1000 \
                            and hparams['use_post_glow']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        run_model = lambda: self.model(
            (sample['word_tokens'], sample['txt_tokens']),
            sample['pron_modified'],
            (None, None, None),
            ph2word=sample['ph2word'],
            word_len=sample['word_lengths'].max(),
            dict_msg=(sample["keys"], sample["values"], sample["key_map"], sample["pinyin"], sample["pinyin_map"]),
            infer=True,
            forward_post_glow=forward_post_glow,
            spk_embed=spk_embed,
            two_stage=hparams['two_stage'],
            mel2word=sample['mel2word'] if hparams['profile_infer'] else None,
            # mel2ph=sample['mel2ph'],
        )

        if hparams['profile_infer']:
            with utils.Timer('model', enable=True):
                outputs = run_model()
            if 'gen_wav_time' not in self.stats:
                self.stats['gen_wav_time'] = 0
            wav_time = float(outputs["mels_out"].shape[1]) * hparams['hop_size'] / hparams["audio_sample_rate"]
            self.stats['gen_wav_time'] += wav_time
            print(f'[Timer] wav total seconds: {self.stats["gen_wav_time"]}')
            from pytorch_memlab import LineProfiler
            with LineProfiler(self.model.forward) as prof:
                run_model()
            prof.print_stats()
        else:
            outputs = run_model()
            sample['outputs'] = outputs['mel_out']
            sample['pron_attn'] = outputs['pron_attn']
            if not hparams.get('infer_post_glow', True):
                sample['outputs'] = outputs['mel_out_fvae']
            if hparams.get('save_attn', False):
                gen_dir = os.path.join(
                    hparams['work_dir'],
                    f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
                os.makedirs(f'{gen_dir}/attn', exist_ok=True)
                item_name = sample['item_name'][0]
                attn = outputs['attn'][0]
                attn = attn.cpu().numpy()
                np.save(f'{gen_dir}/attn/{item_name}.npy', attn)
            return self.after_infer(sample)
    
    def after_infer(self, predictions, sil_start_frame=0):
        predictions = utils.unpack_dict_to_list(predictions)
        assert len(predictions) == 1, 'Only support batch_size=1 in inference.'
        prediction = predictions[0]
        prediction = utils.tensors_to_np(prediction)
        item_name = prediction.get('item_name')
        text = prediction.get('text')
        ph_tokens = prediction.get('txt_tokens')
        mel_gt = prediction["mels"]
        mel2ph_gt = prediction.get("mel2ph")
        mel2ph_gt = mel2ph_gt if mel2ph_gt is not None else None
        mel_pred = prediction["outputs"]
        mel2ph_pred = prediction.get("mel2ph_pred")
        f0_gt = prediction.get("f0")
        f0_pred = prediction.get("f0_pred")

        str_phs = None
        if self.phone_encoder is not None and 'txt_tokens' in prediction:
            str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)

        if 'encdec_attn' in prediction:
            encdec_attn = prediction['encdec_attn']
            encdec_attn = encdec_attn[encdec_attn.max(-1).sum(-1).argmax(-1)]
            txt_lengths = prediction.get('txt_lengths')
            encdec_attn = encdec_attn.T[:txt_lengths, :len(mel_gt)]
        else:
            encdec_attn = None

        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        wav_pred[:sil_start_frame * hparams['hop_size']] = 0
        gen_dir = self.gen_dir
        base_fn = f'[{self.results_id:06d}][{item_name.replace("%", "_")}][%s]'
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        if not hparams['profile_infer']:
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            if hparams.get('save_mel_npy', False):
                os.makedirs(f'{gen_dir}/npy', exist_ok=True)
            if 'encdec_attn' in prediction:
                os.makedirs(f'{gen_dir}/attn_plot', exist_ok=True)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred, encdec_attn]))

            if mel_gt is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph_gt]))

                if hparams['save_f0']:
                    import matplotlib.pyplot as plt
                    f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                    f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                    fig = plt.figure()
                    plt.plot(f0_pred_, label=r'$\hat{f_0}$')
                    plt.plot(f0_gt_, label=r'$f_0$')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                    plt.close(fig)
            print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        self.results_id += 1

        # Dict attention related
        pron_attn = prediction["pron_attn"]
        tokens_pinyin = prediction['pinyin'].squeeze()
        with open(hparams['binary_data_dir']+"/pinyin_encoder.pkl", 'rb') as f:
            pinyin_encoder = pickle.load(f)
        pron_attn = torch.Tensor(pron_attn)
        value, max_idx = pron_attn.max(dim=-1)
        pinyin_tokens = []
        for i in range(1, len(tokens_pinyin)-1):
            for item in tokens_pinyin[i][max_idx[i]:max_idx[i]+2]:
                pinyin_tokens.append(pinyin_encoder[item])
        return {
            'item_name': item_name,
            'text': text.replace(',', '，').replace('.', '。'),
            'pinyin_tokens': " ".join(pinyin_tokens),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
        }