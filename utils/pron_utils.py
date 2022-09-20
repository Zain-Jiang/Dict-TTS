import os
import sys
from pypinyin import pinyin
from pypinyin.contrib.tone_convert import to_initials, to_finals_tone3



checkpoints_dir = 'checkpoints'
compare_dirs = [
                # 'ps_bertfree_aishell3_local3_shallow/generated_30000_',
                'ps_bertfree_aishell3_combine/generated_18000_',
                # 'ps_bertfree_aishell3_combine/generated_104000_',
                # 'ps_bertfree_aishell3_local5/generated_132000_',
                # 'ps_bertfree_aishell3_2/generated_150000_',
                'ps_origin_aishell3/generated_300000_']
                # 'ps_bertfree_aishell3/generated_150000_',
                # 'ps_bertfree_aishell3_2/generated_150000_']

# Get label
def get_label(path, id2labels):
    with open(path, 'r') as f:
        tmp_labels = {}
        lines = f.readlines() 
        for line in lines:
            pron = []
            wav_id, content = line.split('\t')[0], line.split('\t')[1][:-1].split(' ')
            for idx in range(len(content)):
                if idx % 2 == 1:
                    shengmu = to_initials(content[idx], strict=False)
                    yunmu = to_finals_tone3(content[idx].replace('5', ''), strict=False)
                    pron.append(shengmu+yunmu)
            tmp_labels[wav_id] = pron
    id2labels.update(tmp_labels)
    return id2labels

def get_pron_err(line, text, wav_id, id2labels):

    sheng_mu_table = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'ɡ', 'k', \
                        'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', \
                        's', 'y', 'w']
    
    is_error = False
    total_error_num = 0.0
    content = line.replace('<UNK> ', "").split(' ')
    pron = []
    shengmu = ''
    yunmu = ''
    for idx in range(len(content)):
        tone3 = to_finals_tone3(content[idx], strict=False)
        if tone3 in sheng_mu_table or tone3=='':
            shengmu = tone3
        else:
            yunmu = tone3
            if idx > 0 and to_finals_tone3(content[idx-1], strict=False) in sheng_mu_table:
                pron.append(shengmu+yunmu)
            else:
                pron.append(yunmu)
    
    heteronym_num = 0
    text = "".join(text[0])[1:-1]
    heteronym = pinyin(text, heteronym=True)
    for item in heteronym:
        if len(item) > 1:
            heteronym_num += 1
    
    for idx in range(len(id2labels[wav_id])):
        if idx < len(pron):
            pred_item = pron[idx]
            label_item = id2labels[wav_id][idx]
            if pred_item != label_item:
                is_error = True
                total_error_num += 1
                print(pred_item)
                print(label_item)
        else:
            total_error_num += 1
    if is_error:
        print(id2labels[wav_id])
        print(pron)

    return total_error_num, heteronym_num