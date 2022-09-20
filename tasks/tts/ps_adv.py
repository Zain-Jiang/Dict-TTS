import os
import numpy as np
import torch
from torch import nn
import utils
from modules.fastspeech.multi_window_disc import Discriminator
from modules.portaspeech.model import PortaSpeech
from tasks.tts.ps_flow import PortaSpeechFlowTask
from data_gen.tts.data_gen_utils import get_pitch
from utils.common_schedulers import RSQRTSchedule
from utils.hparams import hparams


class PortaSpeechAdvTask(PortaSpeechFlowTask):
    def build_model(self):
        self.model = PortaSpeech(self.phone_encoder)
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

        # Get average absolute duration error, Start
        outputs = run_model()
        dur = outputs['dur']
        src_padding = outputs['word_encoder_out'].data.abs().sum(-1) == 0
        if hparams['dur_scale'] == 'log':
            dur = dur.exp() - 1
        dur = torch.clamp(torch.round(dur), min=0).long()
        pred_mel2word = self.model.length_regulator(dur, (1 - src_padding.long()).sum(-1))[..., 0].detach()
        mel2word = sample['mel2word']
        for i in range(1, mel2word.max() + 1):
            word_dur_err = np.abs(mel2word[mel2word==i].size(0) - pred_mel2word[pred_mel2word==i].size(0))
            if not hasattr(self, 'total_word_durerr'):
                self.total_word_durerr = word_dur_err
                self.total_sentence_durerr = 0
                self.total_words = 0
            else:
                self.total_word_durerr += word_dur_err
        self.total_sentence_durerr += np.abs(mel2word.size(1) - pred_mel2word.size(1))
        self.total_words += mel2word.max() - 1

        if batch_idx == hparams['test_num'] - 2: # Since biaobei has 199 test sample, -2 here
            print("Average Word Duration Error: ")
            print(self.total_word_durerr * (hparams['hop_size'] / hparams['audio_sample_rate']) / self.total_words)
            print("Average Sentence Duration Error: ")
            print(self.total_sentence_durerr)
            print(self.total_sentence_durerr * (hparams['hop_size'] / hparams['audio_sample_rate']) / 200)
        # Get average absolute duration error, End

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
            os.makedirs(f'{gen_dir}/f0', exist_ok=True)
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
                f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                np.save(f'{gen_dir}/f0/{item_name}.npy', f0_pred_)
                np.save(f'{gen_dir}/f0/{item_name}_gt.npy', f0_gt_)
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
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.phone_encoder.decode(ph_tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
        }

    def configure_optimizers(self):
        optimizer_gen = torch.optim.AdamW(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None
        self.scheduler = self.build_scheduler({
            'gen': optimizer_gen,
            'disc': optimizer_disc
        })
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": RSQRTSchedule(optimizer['gen']),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer["disc"],
                **hparams["discriminator_scheduler_params"]),
        }

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["discriminator_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            self.scheduler['gen'].step(self.global_step)
        else:
            self.scheduler['disc'].step(max(self.global_step - hparams["disc_start_steps"], 1))
