from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.fastspeech.conformer.conformer import ConformerDecoder, ConformerEncoder
from modules.fastspeech.fast_tacotron import TacotronEncoder, DecoderRNN, Tacotron2Encoder
from modules.fastspeech.speedy_speech.speedy_speech import ConvBlocks, TextConvEncoder
from modules.fastspeech.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, \
    FastspeechEncoder, RefEncoder
from modules.fastspeech.wavenet_decoder import WN
from modules.commons.rel_transformer_encoder import RelTransformerEncoder
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0

FS_ENCODERS = {
    'fft': lambda hp, embed_tokens, d: FastspeechEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
    'tacotron': lambda hp, embed_tokens, d: TacotronEncoder(
        hp['hidden_size'], len(d), hp['hidden_size'],
        K=hp['encoder_K'], num_highways=4, dropout=hp['dropout']),
    'tacotron2': lambda hp, embed_tokens, d: Tacotron2Encoder(len(d), hp['hidden_size']),
    'conv': lambda hp, embed_tokens, d: TextConvEncoder(embed_tokens, hp['hidden_size'], hp['hidden_size'],
                                                        hp['enc_dilations'], hp['enc_kernel_size'],
                                                        layers_in_block=hp['layers_in_block'],
                                                        norm_type=hp['enc_dec_norm']),
    'conformer': lambda hp, embed_tokens, d: ConformerEncoder(embed_tokens, len(d)),
    'rel_fft': lambda hp, embed_tokens, d: RelTransformerEncoder(
        len(d), hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout']),
}

FS_DECODERS = {
    'fft': lambda hp: FastspeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['dec_num_heads']),
    'rnn': lambda hp: DecoderRNN(hp['hidden_size'], hp['decoder_rnn_dim'], hp['dropout']),
    'conv': lambda hp: ConvBlocks(hp['hidden_size'], hp['hidden_size'], hp['dec_dilations'],
                                  hp['dec_kernel_size'], layers_in_block=hp['layers_in_block'],
                                  norm_type=hp['enc_dec_norm'], dropout=hp['dropout']),
    'wn': lambda hp: WN(hp['hidden_size'], kernel_size=5, n_layers=hp['dec_layers']),
    'conformer': lambda hp: ConformerDecoder(hp['hidden_size']),
}


class FastSpeech2(nn.Module):
    def __init__(self, dictionary, out_dims=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins']
        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_spk_id']:
            self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        elif hparams['use_spk_embed']:
            self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)
        if hparams['use_ref_enc']:
            self.ref_encoder = RefEncoder(
                hparams['audio_num_mel_bins'], hparams['ref_hidden_stride_kernel'],
                ref_norm_layer=hparams['ref_norm_layer'])

        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'], padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, self.padding_idx)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=5, dropout_rate=0.1, odim=2,
                padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])
        if hparams['dec_inp_add_noise']:
            self.z_channels = hparams['z_channels']
            self.dec_inp_noise_proj = nn.Linear(self.hidden_size + self.z_channels, self.hidden_size)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, ref_mels=None,
                f0=None, uv=None, energy=None, skip_decoder=False, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add spk embed
        if hparams['use_spk_embed'] or hparams['use_spk_id']:
            spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        else:
            spk_embed = 0
        if hparams['use_ref_enc']:
            ref_embed = self.ref_encoder(ref_mels)[:, None, :]
            spk_embed = spk_embed + ref_embed

        # add dur
        dur_inp = (encoder_out + spk_embed) * src_nonpadding
        mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = self.expand_states(encoder_out, mel2ph)

        # add pitch and energy embed
        pitch_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        if hparams['use_pitch_embed']:
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out)

        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        if skip_decoder:
            return ret

        if hparams['mel_loss_no_noise']:
            decoder_inp_nonoise = torch.cat([decoder_inp, torch.zeros_like(decoder_inp)], -1)
            decoder_inp_nonoise = self.dec_inp_noise_proj(decoder_inp_nonoise) * tgt_nonpadding
            ret['mel_out_nonoise'] = self.run_decoder(
                decoder_inp_nonoise, tgt_nonpadding, ret, infer=infer, **kwargs)

        if hparams['dec_inp_add_noise']:
            B, T, _ = decoder_inp.shape
            z = kwargs.get('adv_z', torch.randn([B, T, self.z_channels])).to(decoder_inp.device)
            ret['adv_z'] = z
            decoder_inp = torch.cat([decoder_inp, z], -1)
            decoder_inp = self.dec_inp_noise_proj(decoder_inp) * tgt_nonpadding
        ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)

        return ret

    def add_dur(self, dur_input, mel2ph, txt_tokens, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        if hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if mel2ph is None:
            dur, xs = self.dur_predictor.inference(dur_input, src_padding)
            ret['dur'] = xs
            ret['dur_choice'] = dur
            mel2ph = self.length_regulator(dur, (1 - src_padding.long()).sum(-1))[..., 0].detach()
        else:
            ret['dur'] = self.dur_predictor(dur_input, src_padding)
        if mel2ph.shape[1] % hparams['frames_multiple'] > 0:
            max_frames = mel2ph.shape[1] // hparams['frames_multiple'] * hparams['frames_multiple']
            mel2ph = mel2ph[:, :max_frames]
            # pad_len = hparams['frames_multiple'] - mel2ph.shape[1] % hparams['frames_multiple']
            # mel2ph = torch.cat([mel2ph] + [mel2ph[:, -1:]] * pad_len, -1)
        ret['mel2ph'] = mel2ph
        return mel2ph

    def add_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        if hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, hparams, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            hparams, pitch_padding=pitch_padding)
        if hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def run_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding

    def out2mel(self, out):
        return out

    def out2dur(self, xs):
        xs = xs.squeeze(-1)  # (B, Tmax)
        dur = torch.clamp(torch.round(xs.exp() - 1), min=0).long()  # avoid negative value
        return dur

    def expand_states(self, h, mel2ph):
        h = F.pad(h, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, h.shape[-1]])
        h = torch.gather(h, 1, mel2ph_)  # [B, T, H]
        return h
