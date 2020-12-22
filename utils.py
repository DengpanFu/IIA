#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os, sys
import numpy as np
import time, pickle
import scipy.io as sio
import torch
from torch.nn import functional as F

def load_data(path, dtype='torch', mat_type=False):
    if not mat_type:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = sio.loadmat(path)
    q_feat, q_pid, q_cam = data['q_feat'], data['q_id'], data['q_cam']
    g_feat, g_pid, g_cam = data['g_feat'], data['g_id'], data['g_cam']
    if dtype == 'torch':
        q_feat, g_feat = torch.from_numpy(q_feat), torch.from_numpy(g_feat)
    out = [q_feat, q_pid, q_cam, g_feat, g_pid, g_cam]
    return out

def pairwise_distance(x, y, dist_type='cosine'):
    """ Calculate pairwise distance """
    if dist_type == 'euclidean':
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
               torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
    elif dist_type == 'cosine':
        x = F.normalize(x)
        y = F.normalize(y)
        dist = torch.mm(x, y.t()).mul_(-1).add(1)
    else:
        raise TypeError("Unknown dist_type={}.".format(dist_type))
    return dist

def mah_dist(x, y, M=None):
    """ Calculate Mahalanobis Distance if Matrix `M` provided, 
        Or calculate pairwise Euclidean distance
    """
    if M is None:
        return pairwise_distance(x, y, 'euclidean')
    u = (x.matmul(M)*x).sum(1)
    v = (y.matmul(M)*y).sum(1)
    uv = x.matmul(M).matmul(y.t())
    return u.view(-1, 1) + v.view(1, -1) - 2 * uv

def tensor2numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x

def numpy2tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    return x

def print_scores(mAP, cmc_scores, p_str=''):
    if p_str:
        print('{:<15}:'.format(p_str), end='')
    print(('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}],' 
        ' [cmc10: {:5.2%}]').format(mAP, *cmc_scores[[0, 4, 9]]))

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.where(mask)[0]

    cmc[rows_good[0]:] = 1
    d_recall = 1.0/ngood
    precision = (np.arange(len(rows_good), dtype=np.float) + 1) / (rows_good + 1)
    if rows_good[0] == 0:
        old_precision = np.ones(len(rows_good))
        old_precision[1:] = np.arange(1, len(rows_good), dtype=np.float) / rows_good[1:]
    else:
        old_precision = np.arange(len(rows_good), dtype=np.float) / rows_good
    ap = np.sum((precision + old_precision) / 2. * d_recall)

    return ap, cmc

def compute_score(dist, q_id, q_cam, g_id, g_cam, verbose=True, out_aps=False):
    dist = tensor2numpy(dist)
    q_id, q_cam = tensor2numpy(q_id), tensor2numpy(q_cam)
    g_id, g_cam = tensor2numpy(g_id), tensor2numpy(g_cam)
    t1 = time.time()
    cmc = torch.IntTensor(len(g_id)).zero_()
    ap = 0.0
    aps = []
    for i in range(len(q_id)):
        index = dist[i].argsort()
        ql, qc, gl, gc = q_id[i], q_cam[i], g_id, g_cam

        good_index = np.where((gl==ql) & (gc!=qc))[0]
        junk_index = np.where(((gl==ql) & (gc==qc)) | (gl==-1))[0]

        ap_tmp, cmc_tmp = compute_mAP(index, good_index, junk_index)
        aps.append(ap_tmp)
        if cmc_tmp[0]==-1:
            continue
        cmc = cmc + cmc_tmp
        ap += ap_tmp
        if verbose and i % 500 == 0:
            print("Precessing [{:d}/{:d}], Using: {:.3f}s ...".format(
                i+1, len(q_id), time.time() - t1))

    cmc = cmc.float()
    cmc = cmc/len(q_id) #average cmc
    ap = ap/len(q_id)
    if not out_aps:
        return ap, cmc
    else:
        return ap, cmc, aps

