import math
from re import X
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.onnx.operators

from utils.hparams import hparams
from modules.commons.rel_transformer_encoder import Encoder, sequence_mask
from modules.dict_tts.layers.utils import split_heads, combine_heads, mask_logits, mask_weights_attn, add_pron_rule, gumbel_softmax

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class S2PAAttention(nn.Module):
    def __init__(self, query_size=192, key_size=768, value_size=768, num_heads=1, dropout_rate=0.1):
        super(S2PAAttention, self).__init__()
        self.q_transform = nn.Linear(query_size, query_size, bias=False)
        self.k_transform = nn.Linear(key_size, query_size, bias=False)
        self.v_transform = nn.Linear(value_size, query_size, bias=False)
        self.output_transform = nn.Linear(query_size, query_size, bias=False)
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout_rate)

        self.pinyin_embedding = nn.Embedding(hparams['value_embedding_size'], query_size, padding_idx=0)

    def forward(self, x, dict_msg, pron_modified, bias=None):
        keys, values, key_map, pinyin, pinyin_map = dict_msg[0], dict_msg[1], dict_msg[2], dict_msg[3], dict_msg[4]

        q = self.q_transform(x.transpose(1,2))
        k = self.k_transform(keys)
        k = k.reshape([k.shape[0], -1, k.shape[-1]])
        v = self.v_transform(values)
        v = v.reshape([v.shape[0], -1, v.shape[-1]])
        q = split_heads(q, self.num_heads) # [batch, num_heads, length_q, depth_k']
        k = split_heads(k, self.num_heads) # [batch, num_heads, length_q * length_k, depth_k']
        k = k.reshape([k.shape[0], self.num_heads, q.shape[2], -1, k.shape[-1]])
        v = split_heads(v, self.num_heads) # [batch, num_heads, length_q * length_k, depth_v']
        v = v.reshape([v.shape[0], self.num_heads, q.shape[2], -1, v.shape[-1]])
        key_depth_per_head = self.key_size // self.num_heads
        q = q * key_depth_per_head ** -0.5
        logits = torch.matmul(k, q.unsqueeze(-1)).squeeze(-1) # [batch, num_heads, length_q, length_k]
        if bias is not None:
            logits += bias
        logits = mask_logits(logits, key_map)
        weights = F.softmax(logits, dim=-1)
        align = weights.permute(0, 1, 3, 2)
        weights = self.attn_dropout(weights)
        context = torch.matmul(weights.unsqueeze(-2), v).squeeze(-2)

        context = combine_heads(context)
        context = self.output_transform(context).transpose(1,2)

        # Pronunciation module (without Gumbel Softmax)
        pinyin = self.pinyin_embedding(pinyin)
        pron_weights = mask_weights_attn(weights.squeeze(1), pinyin, (key_map,pinyin_map))
        if hparams['language'] == 'zh':
            pron_weights = add_pron_rule(pron_weights, pinyin_map, pron_modified)
        pron = torch.matmul(pron_weights.unsqueeze(-2), pinyin).squeeze(-2).transpose(1,2)

        return context, align, pron, pron_weights


class S2PATextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 window_size=None,
                 block_length=None,
                 gin_channels=0,
                 pre_ln=True,
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
        self.gin_channels = gin_channels
        if n_vocab > 0:
            self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=0)
            nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
            self.word_emb = nn.Embedding(hparams['word_size'], hidden_channels, padding_idx=0)
            nn.init.normal_(self.word_emb.weight, 0.0, hidden_channels ** -0.5)

        self.semantic_encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            pre_ln=pre_ln,
        )

        self.s2pa_attention = S2PAAttention(query_size=hidden_channels)

        self.linguistic_encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            pre_ln=pre_ln,
        )

    def forward(self, txt_tokens, dict_msg, pron_modified):
        word_tokens, ph_tokens = txt_tokens[0], txt_tokens[1]
        x_lengths = (word_tokens > 0).long().sum(-1)
        x = self.word_emb(word_tokens) * math.sqrt(self.hidden_channels)  # [b, t, h]

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.semantic_encoder(x, x_mask)
        
        context, dict_attn, pron, pron_align = self.s2pa_attention(x, dict_msg, pron_modified)
        context = context * x_mask
        x = context + pron
        x = self.linguistic_encoder(x, x_mask)

        return x.transpose(1, 2), dict_attn, pron_align, context.transpose(1, 2)

class DictEncoder(nn.Module):
    def __init__(self, dictionary, hidden_size=None, num_layers=None, kernel_size=None, num_heads=None, norm='ln'):
        super().__init__()
        num_heads = hparams['num_heads'] if num_heads is None else num_heads
        hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = hparams['dropout']
        self.dictionary = dictionary

        # Word embedding
        self.S2PA_module = S2PATextEncoder(
            len(self.dictionary), self.hidden_size, self.hidden_size,
            self.hidden_size * 4, hparams['num_heads'], 2,
            hparams['enc_ffn_kernel_size'], hparams['dropout'])


    def forward(self, txt_tokens, pron_modified, key_value_map, dict_msg, ph2word, spk_embed=None, padding_mask=None, attn_mask=None, return_hiddens=False):

        # Obtain the semantics of word-level input
        txt_nonpadding = (txt_tokens[0] > 0).float()[:, :, None]
        x, dict_attn, pron_attn, context = self.S2PA_module(txt_tokens, dict_msg, pron_modified)
        x = x * txt_nonpadding

        return x, dict_attn, pron_attn, context