import math

import torch
import torch.nn.functional as F
from torch import nn

import utils
from modules.dict_tts.fvae_semantics import FVAE_semantics
from modules.dict_tts.layers.dict_encoder import DictEncoder
from modules.portaspeech.model import PortaSpeech
from utils.hparams import hparams


class PortaSpeech_dict(PortaSpeech):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary)
        self.dict_encoder = DictEncoder(
            dictionary, self.hidden_size, hparams['word_enc_layers'], 1, num_heads=hparams['num_heads'])
        self.fvae = FVAE_semantics(
            in_out_channels=self.out_dims,
            hidden_channels=hparams['fvae_enc_dec_hidden'], latent_size=hparams['latent_size'],
            kernel_size=hparams['fvae_kernel_size'],
            enc_n_layers=hparams['fvae_enc_n_layers'],
            dec_n_layers=hparams['fvae_dec_n_layers'],
            gin_channels=self.hidden_size,
            use_prior_glow=hparams['use_prior_glow'],
            glow_hidden=hparams['prior_glow_hidden'],
            glow_kernel_size=hparams['glow_kernel_size'],
            glow_n_blocks=hparams['prior_glow_n_blocks'],
            strides=[4]
        )
        del self.ph_encoder
        del self.word_encoder


    def forward(self, txt_tokens, pron_modified, key_value_map, ph2word, word_len, dict_msg, mel2word=None, mel2ph=None, spk_embed=None,
                infer=False, tgt_mels=None, forward_post_glow=True, two_stage=True):

        is_training = self.training
        if forward_post_glow and two_stage:
            self.eval()
        ret = {}

        if hparams['use_spk_embed'] or hparams['use_spk_id']:
            spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]  # speaker embedding
        else:
            spk_embed = 0

        padding_mask = txt_tokens[0].eq(0)
        with utils.Timer('encoder', enable=hparams['profile_infer']):
            x, tgt_nonpadding = self.run_text_encoder(txt_tokens, pron_modified, key_value_map, dict_msg, ph2word, word_len, mel2word, mel2ph, ret,  spk_embed=spk_embed, padding_mask=padding_mask)
        
        x = x * tgt_nonpadding
        ret['x_mask'] = tgt_nonpadding
        ret['decoder_inp'] = x

        with utils.Timer('fvae', enable=hparams['profile_infer']):
            ret['mel_out_fvae'] = ret['mel_out'] = self.run_decoder(x, tgt_nonpadding, ret, infer, tgt_mels)
        if (forward_post_glow or not two_stage) and hparams['use_post_glow']:
            with utils.Timer('post_glow', enable=hparams['profile_infer']):
                self.run_post_glow(tgt_mels, infer, is_training, ret)
        return ret
    
    def add_dur(self, dur_input, mel2word, ret, **kwargs):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = dur_input.data.abs().sum(-1) == 0
        dur_input = dur_input.detach() + hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ret['dur'] = dur
        if mel2word is None:
            if hparams['dur_scale'] == 'log':
                dur = dur.exp() - 1
            dur = torch.clamp(torch.round(dur), min=0).long()
            mel2word = self.length_regulator(dur, (1 - src_padding.long()).sum(-1))[..., 0].detach()
        return mel2word

    def run_text_encoder(self, txt_tokens, pron_modified, key_value_map, dict_msg, ph2word, word_len, mel2word, mel2ph, ret, spk_embed=None, padding_mask=None):
        nonpadding_mask = (1-padding_mask.float())[:,:,None]
        with utils.Timer('dict_encoder', enable=hparams['profile_infer']):
            word_encoder_out, dict_attn, pron_attn, context = self.dict_encoder(txt_tokens, pron_modified, key_value_map, dict_msg, ph2word, spk_embed=spk_embed, padding_mask=padding_mask)
            ret['dict_attn'] = dict_attn
            ret['rel'] = None
            ret['dp_attn'] = None
            ret['pron_attn'] = pron_attn
            ret['synta'] = torch.zeros_like(word_encoder_out).to(word_encoder_out.device)
            # ret['synta'] = synta_feats
        word_encoder_out = word_encoder_out + spk_embed

        dur_input = (word_encoder_out) * nonpadding_mask
        mel2word = self.add_dur(dur_input, mel2word, ret, word_len=word_len)
        if mel2word.shape[1] % hparams['frames_multiple'] > 0:
            pad_len = hparams['frames_multiple'] - mel2word.shape[1] % hparams['frames_multiple']
            mel2word = torch.cat([mel2word] + [mel2word[:, -1:]] * pad_len, -1)
        tgt_nonpadding = (mel2word > 0).float()[:, :, None]
        ret['word_encoder_out'] = word_encoder_out
        x = word_encoder_out

        x = F.pad(x, [0, 0, 1, 0])
        mel2word_ = mel2word[..., None].repeat([1, 1, x.shape[-1]])
        x = torch.gather(x, 1, mel2word_)  # [B, T, H]
        ret['synta'] = F.pad(ret['synta'], [0,0,1,0])
        ret['synta'] = torch.gather(ret['synta'], 1, mel2word_)
        return x, tgt_nonpadding
    
    def run_decoder(self, x, tgt_nonpadding, ret, infer, tgt_mels=None):
        x = x.transpose(1, 2)  # [B, H, T]
        ret['synta'] = ret['synta'].transpose(1, 2)
        tgt_nonpadding = tgt_nonpadding.transpose(1, 2)  # [B, H, T]
        if infer:
            mel_out, ret['z_p'] = self.fvae(g=x, infer=True, semantics=ret['synta'])
        else:
            tgt_mels = tgt_mels.transpose(1, 2)  # [B, 80, T]
            mel_out, ret['kl'], ret['z_p'], ret['m_q'], ret['logs_q'] = \
                self.fvae(tgt_mels, tgt_nonpadding, g=x, semantics=ret['synta'])
        return mel_out.transpose(1, 2)