import os
import torch
import torch.nn.functional as F
from torch import nn
import utils
from modules.portaspeech.model import PortaSpeech
from modules.fastspeech.tts_modules import mel2ph_to_dur
from tasks.tts.fs2 import FastSpeech2Task
from tasks.tts.dataset_utils import FastSpeechWordDataset
from utils.common_schedulers import RSQRTSchedule
from utils.hparams import hparams
from utils.plot import spec_to_figure, dur_to_figure
from utils.tts_utils import get_diagonal_focus_rate, get_focus_rate, get_phone_coverage_rate
from vocoders.base_vocoder import get_vocoder_cls


class PortaSpeechFlowTask(FastSpeech2Task):
    def __init__(self):
        super().__init__()
        self.dataset_cls = FastSpeechWordDataset

    def build_model(self):
        self.model = PortaSpeech(self.phone_encoder)
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
            if hparams['glow_type'] == 'v1':
                from modules.glow.glow_modules import Glow
            if hparams['glow_type'] == 'v3':
                from modules.glow.glow_v3_modules import GlowV3 as Glow
            cond_hs = 80
            if hparams.get('use_txt_cond', True):
                cond_hs = cond_hs + hparams['hidden_size']
            if hparams.get('use_latent_cond', False):
                cond_hs = cond_hs + hparams['latent_size']
            if hparams['use_g_proj']:
                self.g_proj = nn.Conv1d(cond_hs, 160, 5, padding=2)
                cond_hs = 160
            self.model.post_flow = Glow(
                80, hparams['post_glow_hidden'], hparams['post_glow_kernel_size'], 1,
                hparams['post_glow_n_blocks'], hparams['post_glow_n_block_layers'],
                n_split=4, n_sqz=2,
                gin_channels=cond_hs,
                share_cond_layers=hparams['post_share_cond_layers'],
                share_wn_layers=hparams['share_wn_layers'],
                sigmoid_scale=hparams['sigmoid_scale']
            )
        utils.print_arch(self.model)
        return self.model

    def on_train_start(self):
        super(PortaSpeechFlowTask, self).on_train_start()
        for n, m in self.model.named_children():
            utils.num_params(m, model_name=n)
        if hasattr(self.model, 'fvae'):
            for n, m in self.model.fvae.named_children():
                utils.num_params(m, model_name=f'fvae.{n}')

    def _training_step(self, sample, batch_idx, opt_idx):
        training_post_glow = self.global_step >= hparams['post_glow_training_start'] and hparams['use_post_glow']
        if hparams['two_stage'] and \
                ((opt_idx == 0 and training_post_glow) or (opt_idx == 1 and not training_post_glow)):
            return None
        loss_output = self.run_model(self.model, sample, training_post_glow=training_post_glow)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output

    def run_model(self, model, sample, return_output=False, training_post_glow=False):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = self.model(sample['txt_tokens'],
                            ph2word=sample['ph2word'],
                            mel2word=sample['mel2word'],
                            mel2ph=sample['mel2ph'],
                            word_len=sample['word_lengths'].max(),
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
        if not return_output:
            return losses
        else:
            return losses, output

    def add_dur_loss(self, dur_pred, mel2word, word_len, txt_tokens, losses=None):
        T = word_len.max()
        dur_gt = mel2ph_to_dur(mel2word, T).float()
        nonpadding = (torch.arange(T).to(dur_pred.device)[None, :] < word_len[:, None]).float()
        dur_pred = dur_pred * nonpadding
        dur_gt = dur_gt * nonpadding
        if hparams['dur_scale'] == 'log':
            dur_gt = (dur_gt + 1).log()
        wdur = F.l1_loss(dur_pred, dur_gt, reduction='none')
        wdur = (wdur * nonpadding).sum() / nonpadding.sum()
        losses['wdur'] = wdur
        if hparams['lambda_sent_dur'] > 0:
            assert hparams['dur_scale'] == 'linear'
            sdur = F.l1_loss(dur_pred.sum(-1), dur_gt.sum(-1))
            losses['sdur'] = sdur * hparams['lambda_sent_dur']

        if not self.training:
            if hparams['dur_scale'] == 'log':
                dur_pred = (dur_pred + 1).log()
                dur_gt = (dur_gt + 1).log()
            if hparams['dur_level'] == 'ph':
                B, T = txt_tokens.shape
                is_sil = torch.zeros_like(txt_tokens).bool()
                for p in self.sil_ph:
                    is_sil = is_sil | (txt_tokens == self.phone_encoder.encode(p)[0])
                is_sil = is_sil.float()  # [B, T_txt]
                word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
                word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
                word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
                wdur_loss = F.l1_loss(word_dur_p, word_dur_g, reduction='none')
                word_nonpadding = (word_dur_g > 0).float()
                wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            else:
                wdur_loss = F.l1_loss(dur_pred, dur_gt, reduction='none')
                word_nonpadding = (dur_gt > 0).float()
                wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.l1_loss(sent_dur_p, sent_dur_g, reduction='mean')
            losses['sdur'] = sdur_loss.mean()

    def get_attn_stats(self, attn, sample, logging_outputs, prefix=''):
        # diagonal_focus_rate
        txt_lengths = sample['txt_lengths'].float()  # - 1 # exclude eos
        mel_lengths = sample['mel_lengths'].float()
        src_padding_mask = sample['txt_tokens'].eq(0)  # | input.eq(self.eos_idx)  # also exclude eos
        target_padding_mask = sample['mels'].abs().sum(-1).eq(0)
        src_seg_mask = sample['txt_tokens'].eq(self.seg_idx)
        attn_ks = txt_lengths.float() / mel_lengths.float()

        focus_rate = get_focus_rate(attn, src_padding_mask, target_padding_mask).mean().data
        phone_coverage_rate = get_phone_coverage_rate(
            attn, src_padding_mask, src_seg_mask, target_padding_mask).mean()
        diagonal_focus_rate, diag_mask = get_diagonal_focus_rate(
            attn, attn_ks, mel_lengths, src_padding_mask, target_padding_mask)
        logging_outputs[f'{prefix}focus_rate'] = focus_rate.mean().data
        logging_outputs[f'{prefix}phone_coverage_rate'] = phone_coverage_rate.mean().data
        logging_outputs[f'{prefix}diagonal_focus_rate'] = diagonal_focus_rate.mean().data

    def plot_dur_ph(self, batch_idx, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2ph_to_dur(sample['mel2ph'], T_txt)[0]
        dur_pred = model_out['dur']
        if hparams['dur_scale'] == 'log':
            dur_pred = dur_pred.exp() - 1
        dur_pred = torch.clamp(torch.round(dur_pred), min=0).long()
        txt = self.phone_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
        txt = txt.split(" ")
        self.logger.add_figure(
            f'dur_{batch_idx}', dur_to_figure(dur_gt, dur_pred, txt, sample['mels'][0]), self.global_step)

    def plot_dur(self, batch_idx, sample, model_out):
        T = sample['word_lengths'].max()
        dur_gt = mel2ph_to_dur(sample['mel2word'], T)[0]
        dur_pred = model_out['dur'][0]
        if hparams['dur_scale'] == 'log':
            dur_pred = dur_pred.exp() - 1
        dur_pred = torch.clamp(torch.round(dur_pred), min=0).long()
        txt = sample['words'][0]
        self.logger.add_figure(
            f'dur_{batch_idx}', dur_to_figure(dur_gt, dur_pred, txt, sample['mels'][0]),
            self.global_step)

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
            model_out = self.model(sample['txt_tokens'],
                                   ph2word=sample['ph2word'],
                                   word_len=sample['word_lengths'].max(),
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
                self.logger.add_audio(f'wav_{batch_idx}', wav_pred, self.global_step, 22050)
                self.logger.add_figure(
                    f'mel_{batch_idx}', spec_to_figure(model_out['mel_out'][0], vmin, vmax), self.global_step)
            wav_pred = self.vocoder.spec2wav(model_out['mel_out_fvae'][0].cpu())
            if self.global_step > 0:
                self.logger.add_audio(f'wav_fvae_{batch_idx}', wav_pred, self.global_step, 22050)
                self.logger.add_figure(
                    f'mel_fvae_{batch_idx}', spec_to_figure(model_out['mel_out_fvae'][0], vmin, vmax), self.global_step)
            if hparams['dur_level'] == 'word':
                if 'attn' in model_out:
                    self.logger.add_figure(
                        f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)
                self.plot_dur(batch_idx, sample, model_out)
            else:
                self.plot_dur_ph(batch_idx, sample, model_out)
        return outputs

    def validation_end(self, outputs):
        self.vocoder = None
        return super().validation_end(outputs)

    def build_optimizer(self, model):
        if hparams['two_stage'] and hparams['use_post_glow']:
            self.optimizer = torch.optim.AdamW(
                [p for name, p in self.model.named_parameters() if 'post_flow' not in name],
                lr=hparams['lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                weight_decay=hparams['weight_decay'])
            self.post_flow_optimizer = torch.optim.AdamW(
                self.model.post_flow.parameters(),
                lr=hparams['post_flow_lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                weight_decay=hparams['weight_decay'])
            return [self.optimizer, self.post_flow_optimizer]
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=hparams['lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                weight_decay=hparams['weight_decay'])
            return [self.optimizer]

    def build_scheduler(self, optimizer):
        return RSQRTSchedule(optimizer[0])

    ############
    # infer
    ############
    def test_start(self):
        super().test_start()
        if hparams['use_post_glow']:
            self.model.post_flow.store_inverse()

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

    def test_step(self, sample, batch_idx):
        forward_post_glow = self.global_step >= hparams['post_glow_training_start'] + 1000 \
                            and hparams['use_post_glow']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        run_model = lambda: self.model(
            sample['txt_tokens'],
            ph2word=sample['ph2word'],
            word_len=sample['word_lengths'].max(),
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
            if not hparams.get('infer_post_glow', True):
                sample['outputs'] = outputs['mel_out_fvae']
            if hparams.get('save_attn', False):
                import numpy as np
                gen_dir = os.path.join(
                    hparams['work_dir'],
                    f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
                os.makedirs(f'{gen_dir}/attn', exist_ok=True)
                item_name = sample['item_name'][0]
                attn = outputs['attn'][0]
                attn = attn.cpu().numpy()
                np.save(f'{gen_dir}/attn/{item_name}.npy', attn)
            return self.after_infer(sample)
