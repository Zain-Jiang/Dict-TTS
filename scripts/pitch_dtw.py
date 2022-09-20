import sys
from glob import glob
import numpy as np
from dtw import dtw as DTW
manhattan_distance = lambda x, y: np.abs(x - y)
# euclidean_distance = lambda x, y: (x - y) ** 2

# checkpoints_name = 'ps_origin_aishell3'
# checkpoints_name = 'ps_bertemb_aishell3'

# checkpoints_name = 'ps_biaobei_bertemb'
# steps = 'generated_150000_'

# checkpoints_name = 'ps_origin_biaobei_adv_2'
# steps = 'generated_208000_'

# checkpoints_name = 'ps_dict_biaobei_768_4'
# steps = 'generated_200000_'

# checkpoints_name = 'ps_wordonly_biaobei'
# steps = 'generated_102000_'

# checkpoints_name = 'ps_biaobei_nlr'
# steps = 'generated_150000_'

checkpoints_name = 'ps_origin_biaobei_g2pm'
steps = 'generated_300000_'

pred_fn = glob(f'checkpoints/{checkpoints_name}/{steps}/f0/*.npy')
gt_fn = glob(f'checkpoints/{checkpoints_name}/{steps}/f0/*_gt.npy')
f0_pred = []
for item in pred_fn:
    if item not in gt_fn:
        f0_pred.append(np.load(item))
f0_gt = []
for item in gt_fn:
    f0_gt.append(np.load(item))

dtw = 0
dtw_list = []
for i in range(len(gt_fn)):
    distance, _, _, _ = DTW(f0_pred[i], f0_gt[i], manhattan_distance)
    distance = distance / len(f0_gt[i])
    dtw += distance
    dtw_list.append(distance)
print(dtw/len(f0_gt))

from scipy.stats import skew as SKEW
from scipy.stats import kurtosis as KURTOSIS
std = 0
skew = 0
kurtosis = 0
dtw_list = []
for i in range(len(gt_fn)):
    std += np.std(f0_pred[i])
    skew += SKEW(f0_pred[i], axis=0, bias=True)
    kurtosis += KURTOSIS(f0_pred[i], axis=0, bias=True)
print(std/len(f0_pred))
print(skew/len(f0_pred))
print(kurtosis/len(f0_pred))