import math

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn

import utils
from modules.commons.common_layers import Embedding, MultiheadAttention
from modules.commons.rel_transformer_encoder import Encoder, sequence_mask
from modules.portaspeech.fvae import FVAE
from modules.fastspeech.tts_modules import LengthRegulator, FastspeechDecoder, LayerNorm
from modules.portaspeech.glow_modules import ConvReluNorm
from modules.portaspeech.utils import group_hidden_by_segs
from utils.hparams import hparams


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """

        :param x: [B, T]
        :return: [B, T, H]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DurationPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding='SAME'):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == 'SAME'
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = nn.Sequential(torch.nn.Linear(n_chans, 1), nn.Softplus())

    def forward(self, xs, x_masks=None):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]
        xs = self.linear(xs.transpose(1, -1))[:, :, 0]  # [B, T, C]
        xs = xs * (1 - x_masks.float())  # (B, T, C)
        return xs


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 window_size=4,
                 block_length=None,
                 prenet=True,
                 gin_channels=0,
                 pre_ln=False,
                 ):

        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.prenet = prenet
        self.gin_channels = gin_channels
        if n_vocab > 0:
            self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=0)
            nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if prenet:
            self.pre = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size=5,
                                    n_layers=3, p_dropout=0.5)
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            pre_ln=pre_ln,
        )

    def forward(self, x, x_mask=None):
        if self.n_vocab > 0:
            x_lengths = (x > 0).long().sum(-1)
            x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        else:
            x_lengths = (x.abs().sum(-1) > 0).long().sum(-1)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)
        return x.transpose(1, 2)


class PortaSpeech(nn.Module):
    def __init__(self, dictionary, out_dims=None):
        super().__init__()
        # configs
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins']
        # build linguistic encoder
        self.ph_encoder = TextEncoder(
            len(self.dictionary), self.hidden_size, self.hidden_size,
            self.hidden_size * 4, hparams['num_heads'], hparams['enc_layers'],
            hparams['enc_ffn_kernel_size'], hparams['dropout'], prenet=True)
        if hparams['dur_level'] == 'word':
            self.word_encoder = FastspeechDecoder(
                self.hidden_size, hparams['word_enc_layers'], 1, num_heads=hparams['num_heads'])
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
            self.enc_pos_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.dec_query_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.dec_res_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.attn = MultiheadAttention(self.hidden_size, 1, encoder_decoder_attention=True, bias=False)
            self.attn.enable_torch_version = False
        if hparams['num_spk'] > 1:
            if hparams['use_spk_id']:
                self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
            elif hparams['use_spk_embed']:
                self.spk_embed_proj = nn.Linear(256, self.hidden_size, bias=True)
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=128,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'], padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        # build VAE decoder
        self.fvae = FVAE(
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
        # build post flow
        self.use_post_glow = use_post_glow = hparams['use_post_glow']
        if use_post_glow:
            from modules.glow.glow_modules import Glow
            cond_hs = 80
            if hparams.get('use_txt_cond', True):
                cond_hs = cond_hs + hparams['hidden_size']
            if hparams.get('use_latent_cond', False):
                cond_hs = cond_hs + hparams['latent_size']
            if hparams['use_g_proj']:
                self.g_proj = nn.Conv1d(cond_hs, 160, 5, padding=2)
                cond_hs = 160
            self.post_flow = Glow(
                80, hparams['post_glow_hidden'], hparams['post_glow_kernel_size'], 1,
                hparams['post_glow_n_blocks'], hparams['post_glow_n_block_layers'],
                n_split=4, n_sqz=2,
                gin_channels=cond_hs,
                share_cond_layers=hparams['post_share_cond_layers'],
                share_wn_layers=hparams['share_wn_layers'],
                sigmoid_scale=hparams['sigmoid_scale']
            )
            self.prior_dist = dist.Normal(0, 1)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, ph2word, word_len, mel2word=None, mel2ph=None, spk_embed=None,
                infer=False, tgt_mels=None, forward_post_glow=True, two_stage=True):
        is_training = self.training
        if forward_post_glow and two_stage:
            self.eval()
        ret = {}

        if hparams['use_spk_embed'] or hparams['use_spk_id']:
            spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]  # speaker embedding 映射
        else:
            spk_embed = 0

        with utils.Timer('encoder', enable=hparams['profile_infer']):
            x, tgt_nonpadding = self.run_text_encoder(txt_tokens, ph2word, word_len, mel2word, mel2ph, ret, spk_embed)

        x = x + spk_embed
        x = x * tgt_nonpadding
        ret['x_mask'] = tgt_nonpadding
        ret['decoder_inp'] = x
        with utils.Timer('fvae', enable=hparams['profile_infer']):
            ret['mel_out_fvae'] = ret['mel_out'] = self.run_decoder(x, tgt_nonpadding, ret, infer, tgt_mels)
        if (forward_post_glow or not two_stage) and hparams['use_post_glow']:
            with utils.Timer('post_glow', enable=hparams['profile_infer']):
                self.run_post_glow(tgt_mels, infer, is_training, ret)
        return ret

    def run_text_encoder(self, txt_tokens, ph2word, word_len, mel2word, mel2ph, ret, spk_embed):
        word2word = torch.arange(word_len)[None, :].to(ph2word.device) + 1  # [B, T_mel, T_word]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        with utils.Timer('ph_encoder', enable=hparams['profile_infer']):
            ret['ph_encoder_out'] = ph_encoder_out = self.ph_encoder(txt_tokens) * src_nonpadding
            ph_encoder_out = ph_encoder_out + spk_embed
        if hparams['dur_level'] == 'word':
            word_encoder_out = group_hidden_by_segs(ph_encoder_out, ph2word, word_len)
            with utils.Timer('word_encoder', enable=hparams['profile_infer']):
                word_encoder_out = self.word_encoder(word_encoder_out)
            mel2word = self.add_dur(ph_encoder_out, mel2word, ret, ph2word=ph2word, word_len=word_len)
            if mel2word.shape[1] % hparams['frames_multiple'] > 0:
                pad_len = hparams['frames_multiple'] - mel2word.shape[1] % hparams['frames_multiple']
                mel2word = torch.cat([mel2word] + [mel2word[:, -1:]] * pad_len, -1)

            tgt_nonpadding = (mel2word > 0).float()[:, :, None]
            enc_pos = self.build_pos_embed(word2word, ph2word)  # [B, T_ph, H]
            dec_pos = self.build_pos_embed(word2word, mel2word)  # [B, T_mel, H]
            dec_word_mask = self.build_word_mask(mel2word, ph2word)  # [B, T_mel, T_ph]
            x, weight = self.attention(ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask)
            ret['attn'] = weight
            ret['word_encoder_out'] = word_encoder_out

            # B, T_ph = ph2word.shape
            # x = torch.zeros([B, word_len.max() + 1, self.hidden_size]).to(ph2word.device).scatter_add(1, ph2word[:,:,None].repeat(1,1,self.hidden_size), ph_encoder_out)
            # x = x[:, 1:]
            # x = F.pad(x, [0, 0, 1, 0]).gather(
            # 1, mel2word[:, :, None].repeat(1, 1, self.hidden_size))
        else:
            mel2ph = self.add_dur(ph_encoder_out, mel2ph, ret)
            if mel2ph.shape[1] % hparams['frames_multiple'] > 0:
                pad_len = hparams['frames_multiple'] - mel2ph.shape[1] % hparams['frames_multiple']
                mel2ph = torch.cat([mel2ph] + [mel2ph[:, -1:]] * pad_len, -1)
            decoder_inp = F.pad(ph_encoder_out, [0, 0, 1, 0])
            mel2ph_ = mel2ph[..., None].repeat([1, 1, ph_encoder_out.shape[-1]])
            x = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]
            tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        return x, tgt_nonpadding

    def attention(self, ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask):
        ph_kv = self.enc_pos_proj(torch.cat([ph_encoder_out, enc_pos], -1))
        word_enc_out_expend = F.pad(word_encoder_out, [0, 0, 1, 0]).gather(
            1, mel2word[:, :, None].repeat(1, 1, self.hidden_size))
        dec_q = self.dec_query_proj(torch.cat([word_enc_out_expend, dec_pos], -1))
        x_res = self.dec_res_proj(torch.cat([word_enc_out_expend, dec_pos], -1))
        ph_kv, dec_q = ph_kv.transpose(0, 1), dec_q.transpose(0, 1)
        x, (weight, _) = self.attn(dec_q, ph_kv, ph_kv, attn_mask=(1 - dec_word_mask) * -1e9)
        x = x.transpose(0, 1)
        x = x + x_res
        return x, weight

    def run_decoder(self, x, tgt_nonpadding, ret, infer, tgt_mels=None):
        x = x.transpose(1, 2)  # [B, H, T]
        tgt_nonpadding = tgt_nonpadding.transpose(1, 2)  # [B, H, T]
        if infer:
            mel_out, ret['z_p'] = self.fvae(g=x, infer=True)
        else:
            tgt_mels = tgt_mels.transpose(1, 2)  # [B, 80, T]
            mel_out, ret['kl'], ret['z_p'], ret['m_q'], ret['logs_q'] = \
                self.fvae(tgt_mels, tgt_nonpadding, g=x)
        return mel_out.transpose(1, 2)

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
        if hparams['dur_level'] == 'word':
            ph2word = kwargs['ph2word']
            word_len = kwargs['word_len']
            B, T_ph = ph2word.shape
            dur = torch.zeros([B, word_len.max() + 1]).to(ph2word.device).scatter_add(1, ph2word, dur)
            dur = dur[:, 1:]
        ret['dur'] = dur
        if mel2word is None:
            if hparams['dur_scale'] == 'log':
                dur = dur.exp() - 1
            dur = torch.clamp(torch.round(dur), min=0).long()
            mel2word = self.length_regulator(dur, (1 - src_padding.long()).sum(-1))[..., 0].detach()
        return mel2word

    def run_post_glow(self, tgt_mels, infer, is_training, ret):
        x_recon = ret['mel_out'].transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        if hparams.get('use_txt_cond', True):
            g = torch.cat([g, ret['decoder_inp'].transpose(1, 2)], 1)
        if hparams.get('use_latent_cond', False):
            g_z = ret['z_p'][:, :, :, None].repeat(1, 1, 1, 4).reshape(B, -1, T)
            g = torch.cat([g, g_z], 1)
        if hparams['use_g_proj']:
            g = self.g_proj(g)
        prior_dist = self.prior_dist
        if not infer:
            if is_training:
                self.train()
            x_mask = ret['x_mask'].transpose(1, 2)
            y_lengths = x_mask.sum(-1)
            if hparams['detach_postflow_input']:
                g = g.detach()
            tgt_mels = tgt_mels.transpose(1, 2)
            if hparams['res_mode'] == 1:
                tgt_mels = tgt_mels - x_recon
            z_postflow, ldj = self.post_flow(tgt_mels, x_mask, g=g)
            ldj = ldj / y_lengths / 80
            ret['z_pf'], ret['ldj_pf'] = z_postflow, ldj
            ret['postflow'] = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
        else:
            x_mask = torch.ones_like(x_recon[:, :1, :])
            z_post = prior_dist.sample(x_recon.shape).to(g.device) * hparams['noise_scale']
            x_recon_, _ = self.post_flow(z_post, x_mask, g, reverse=True)
            x_recon = x_recon + x_recon_ if hparams['res_mode'] == 1 else x_recon_
            ret['mel_out'] = x_recon.transpose(1, 2)

    def build_pos_embed(self, word2word, x2word):
        x_pos = self.build_word_mask(word2word, x2word).float()  # [B, T_word, T_ph]
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())  # [B, T_ph, H]
        return x_pos

    @staticmethod
    def build_word_mask(x2word, y2word):
        return (x2word[:, :, None] == y2word[:, None, :]).long()