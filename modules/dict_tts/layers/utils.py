import torch
from torch.autograd import Variable
import torch.nn.functional as F


def split_heads(x, num_heads):
    """ Split heads
    :param x: A tensor with shape [batch, length, channels]
    :param num_heads: An integer
    :returns: A tensor with shape [batch, heads, length, channels / heads]
    """
    assert x.shape[-1] % num_heads == 0, str(x.shape)
    return x.reshape(x.shape[:-1] + (num_heads, x.shape[-1] // num_heads)).permute(0, 2, 1, 3)


def combine_heads(x):
    """ Combine heads
    :param x: A tensor with shape [batch, heads, length, channels]
    :returns: A tensor with shape [batch, length, heads * channels]
    """
    x = x.permute([0, 2, 1, 3])
    return x.reshape(x.shape[:-2] + (x.shape[-1] * x.shape[-2],))


def mask_weights(weights, v_length_masks):
    for item_num in range(weights.size(0)):
        pad_idx = v_length_masks[item_num][len(v_length_masks[item_num])-1]
        for word_num in range(len(v_length_masks[item_num])):
            pad_idx = v_length_masks[item_num][word_num][len(v_length_masks[item_num][word_num])-1]
            weights[item_num, :, word_num+1, pad_idx:] = - 1e10
    return weights


def balance_mask_value(mask, alpha=0):
    balance_coef = mask.sum(dim=-1,keepdim=True)
    balance_coef[balance_coef==0] = 1
    mask = alpha * mask + (1-alpha) * mask / balance_coef
    return mask

def mask_logits(logits, key_map):
    mask = torch.zeros_like(logits)
    # Balancing for the length of gloss
    # for i in range(1, key_map.max().int()+1):
        # mask = mask + balance_mask_value((key_map == i).float())
    mask = key_map.ne(0).float()[:,None,:,:].repeat(1, logits.size(1), 1, 1)
    logits[mask == 0] = - 1e9
    return logits

def mask_weights_attn(weights, v, key_value_map):
    key_map, value_map = key_value_map
    res = torch.zeros(weights.size(0), weights.size(1), v.size(2)).to(weights.device)

    for i in range(1, key_map.max().int()+1):
        merge_value = (weights * (key_map.eq(i).float())).sum(dim=-1, keepdim=True)
        mask = value_map.eq(i).float()
        res += merge_value.repeat(1,1,v.size(2)) * mask

    return res

def mask_weights_attn_gumbel(weights, v, key_value_map):
    key_map, value_map = key_value_map
    res = torch.zeros(weights.size(0), weights.size(1), weights.size(2), v.size(3)).to(weights.device)
    key_map = key_map[:,None,:,:].repeat(1,weights.size(1),1,1)
    value_map = value_map[:,None,:,:].repeat(1,v.size(1),1,1)
    
    merge_value =torch.zeros_like(weights)[:,:,:,:key_map.max().int()]
    for i in range(1, key_map.max().int()+1):
        merge_value[:,:,:,i-1] = (weights * (key_map == i).float()).sum(dim=-1)
    # merge_value = gumbel_softmax(merge_value, temperature=0.8, hard=True)
    merge_value = merge_value.softmax(dim=-1)
    
    for i in range(1, key_map.max().int()+1):
        mask = (value_map == i).float()
        merge_value_tmp = merge_value[:,:,:,i-1].unsqueeze(-1).repeat(1,1,1,v.size(3))
        res[mask==1] = merge_value_tmp[mask==1]
    return res

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

# def gumbel_softmax_sample(logits, temperature):
#     gumbels = (
#         -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
#     )
#     y = logits + gumbels
#     return F.softmax(y / temperature, dim=-1)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y_soft = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y_soft
    else:
        dim = -1
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        y_hard = y_hard - y_soft.detach() + y_soft
        return y_hard

def add_pron_rule(weights, value_map, pron_modified):
    weights_ = weights.clone().detach()
    with torch.no_grad():
        for i in range(1, value_map.max().int()+1):
            weights_[pron_modified==i] = (value_map[pron_modified==i] == i).float()
    weights_ = weights_ - weights.detach() + weights
    return weights_

    