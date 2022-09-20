import os
import pickle
import numpy as np
import torch
from utils.hparams import hparams

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')


LINE_COLORS = ['w', 'r', 'y', 'cyan', 'm', 'b', 'lime']


def spec_to_figure(spec, vmin=None, vmax=None, title=''):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig


def spec_f0_to_figure(spec, f0s, figsize=None):
    max_y = spec.shape[1]
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
        f0s = {k: f0.detach().cpu().numpy() for k, f0 in f0s.items()}
    f0s = {k: f0 / 10 for k, f0 in f0s.items()}
    fig = plt.figure(figsize=(12, 6) if figsize is None else figsize)
    plt.pcolor(spec.T)
    for i, (k, f0) in enumerate(f0s.items()):
        plt.plot(f0.clip(0, max_y), label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.8)
    plt.legend()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt, mels=None, vmin=-5.5, vmax=1):
    dur_gt = dur_gt.cpu().numpy()
    dur_pred = dur_pred.cpu().numpy()
    dur_gt = np.cumsum(dur_gt).astype(int)
    dur_pred = np.cumsum(dur_pred).astype(int)
    fig = plt.figure(figsize=(12, 6))
    for i in range(len(dur_gt)):
        shift = (i % 8) + 1
        plt.text(dur_gt[i], shift * 4, txt[i])
        plt.text(dur_pred[i], 40 + shift * 4, txt[i])
        plt.vlines(dur_gt[i], 0, 40, colors='b')  # blue is gt
        plt.vlines(dur_pred[i], 40, 80, colors='r')  # red is pred
    plt.xlim(0, max(dur_gt[-1], dur_pred[-1]))
    if mels is not None:
        mels = mels.cpu().numpy()
        plt.pcolor(mels.T, vmin=vmin, vmax=vmax)
    return fig


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure(figsize=(12, 8))
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='cwt')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig

def attn_to_figure(attn, sample, pinyin_encoder, vmin=None, vmax=None, title=''):
    tokens_pinyin = sample['values'][0]
    tokens_gloss = sample['tokens_gloss'][0]
    if isinstance(attn, torch.Tensor):
        attn = attn.cpu().numpy()
    fig = plt.figure()
    plt.title(title)
    plt.pcolor(attn, vmin=vmin, vmax=vmax)

    scale_ls = range(len(sample['words'][0]))
    index_ls = sample['words'][0]
    for i in range(len(index_ls)):
        index_ls[i] = index_ls[i]
    _ = plt.xticks(scale_ls,index_ls, fontproperties=myfont)

    # Draw the text
    count = 0
    for i in range(len(index_ls)):
        if index_ls[i] == '<BOS>' or index_ls[i] == '<EOS>':
            count += 1
            continue
        for idx, item in enumerate(tokens_pinyin[count]):
            if item != 0:
                text = [pinyin_encoder[item]]
                plt.text(count, 0+idx, '\n\n\n'.join(text), fontsize=10)
        count += 1
    return fig

def attn_to_figure_unmerged(attn, sample, vmin=None, vmax=None, title=''):
    tokens_gloss = sample['tokens_gloss'][0]
    attn = attn.transpose(0, 1)

    if isinstance(attn, torch.Tensor):
        attn = attn.cpu().numpy()
    fig = plt.figure(figsize=( len(tokens_gloss)*3, len(sample['words'][0])))
    plt.title(title)

    c = plt.pcolor(attn)
    fig.colorbar(c)

    scale_ls = range(len(sample['words'][0]))
    index_ls = sample['words'][0]
    for i in range(len(index_ls)):
        index_ls[i] = index_ls[i]
    _ = plt.yticks(scale_ls,index_ls, fontproperties=myfont)

    # Draw the text
    count = 0
    for i in range(len(index_ls)):
        text = []
        if index_ls[i] == '<BOS>' or index_ls[i] == '<EOS>':
            count += 1
            continue
        w = 0
        for item in tokens_gloss[count-1]:
            if item == "单音":
                for h, word in enumerate(item):
                    plt.text(1+w, count, word+"\n"*h, fontsize=10,  fontproperties=myfont)
            elif item == '<sos>' or item == '<eos>' or item == '<BOS>' or item == '<EOS>':
                for h, word in enumerate(item):
                    plt.text(0+w, count, word+"\n"*h, fontsize=10,  fontproperties=myfont)
                w += 1
            else:
                for h, word in enumerate(item):
                    plt.text(0+w, count, word+"\n"*h, fontsize=10,  fontproperties=myfont)
                w += 1
        count += 1

    # data_tuple = (sample['words'], attn, tokens_gloss)
    # pickle.dump(data_tuple, open(f"plot_{sample['id'][0]}.pkl", 'wb'))
    return fig


from sklearn import manifold, datasets, decomposition
def visualize_embeddings_tsne(embeddings, type):
    x_tsne = manifold.TSNE(n_components=2, random_state=0, verbose=1, perplexity=20).fit_transform(embeddings)
    
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)     
    fig = plt.figure(figsize=(8, 8))
    for i in range(x_norm.shape[0]):
        plt.text(x_norm[i, 0], x_norm[i, 1], type[i], 
             fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    
    return fig

def visualize_embeddings_PCA(embeddings, type):

    x_pca = decomposition.PCA(n_components=2).fit_transform(embeddings)

    x_min, x_max = x_pca.min(0), x_pca.max(0)
    x_norm = (x_pca - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(8, 8))
    for i in range(x_norm.shape[0]):
        plt.text(x_norm[i, 0], x_norm[i, 1], type[i],
             fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])

    return fig