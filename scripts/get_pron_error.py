# -*- coding: utf-8 -*-  
import re
from jiwer import wer
from pypinyin import pinyin

gold = []
word_num = 0
heteronym_num = 0
with open('./scripts/pron_label/label_set0.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        txt, pron_label = line.split(',')[3], line.split(',')[4]

        pron_list = []
        for item in re.split(' \| | \# ', pron_label[6:-6]):
            pron_list.append(item.replace(' ', ''))
            word_num += 1
        gold.append(" ".join(pron_list))

        # calculate the number of heteronyms
        text = "".join(txt)
        heteronym = pinyin(text, heteronym=True)
        for item in heteronym:
            if len(item) > 1:
                heteronym_num += 1
    print(f'Heteronym num: {heteronym_num}')
    print(f'Word num: {word_num}')

# # Read ps_dict
pred = []
with open('./checkpoints/dict_tts_biaobei_wo_gumbel/generated_300000_/meta.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        pron_label = line.split(',')[3].replace('<UNK> ', '').replace('\n', '').split(' ')
        pron = ''
        pron_list = []
        for i, item in enumerate(pron_label):
            if i % 2 == 0:
                pron += item
            else:
                pron += item
                pron_list.append(pron)
                pron = ''
        pred.append(" ".join(pron_list))

print(len(pred))
print(len(gold))
print("PER: ", "%.2f" % (wer(pred,gold) * 100), "%")