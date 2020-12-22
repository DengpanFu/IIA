#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os, sys, time
import numpy as np
import torch
from torch.nn import functional as F
from utils import *

#'market': {'alpha': 0.75, 'topk': 11, 'num': 4, 'tau': 0.1}, 
CONFIG = {
          'duke'  : {'alpha': 0.82, 'topk': 13, 'num':  7, 'tau': 0.1}, 
          'market': {'alpha': 0.82, 'topk': 11, 'num':  6, 'tau': 0.1}, 
          'cuhk03': {'alpha': 0.82, 'topk':  8, 'num': 12, 'tau': 0.1}, 
          'recomm': {'alpha': 0.82, 'topk': 10, 'num':  6, 'tau': 0.1}
          }

class IIA(object):
    def __init__(self, data_name='', alpha=None, topk=None, num=None, 
        tau=None, device=0, online=True, auto_infer_param=True, verbose=True):
        super(IIA, self).__init__()
        self.data_name = self.parse_dname(data_name)
        self.alpha     = self.parse_param('alpha', alpha)
        self.topk      = self.parse_param('topk', topk)
        self.num       = self.parse_param('num', num)
        self.tau       = self.parse_param('tau', tau)

        if isinstance(device, torch.device):
            self.device = device
        elif device < 0 or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.online = online
        self.auto_infer_param = False if data_name in CONFIG else auto_infer_param

        self.verbose = verbose

    def parse_dname(self, name):
        if   'duke'   in name.lower(): data_name = 'duke'
        elif 'market' in name.lower(): data_name = 'market'
        elif 'cuhk03' in name.lower(): data_name = 'cuhk03'
        elif 'recomm' in name.lower(): data_name = 'recomm'
        else: data_name = name.lower()
        return data_name

    def parse_param(self, key, val):
        if not val is None:
            return val
        else:
            if self.data_name in CONFIG:
                dname = self.data_name
            else:
                dname = 'recomm'
            if not key in CONFIG[dname]:
                raise ValueError('Invalid param={} in (alpha, topk, num, tau)'
                    .format(key))
            else:
                return CONFIG[dname][key]

    def infer_param(self, qN):
        """ Infer the Hyper Parameters if not specific data name 
            For               Market1501, DukeMTMC, and CUHK03-NP, 
            the query num is:    3368,      2228    and   1400
            It's just a rough inference.
        Args:
            qN: query num
        """
        if 3368 - 10 < qN < 3368 + 10:
            self.topk = CONFIG['market']['topk']
            self.num  = CONFIG['market']['num']
        elif 2228 - 10 < qN < 2228 + 10:
            self.topk = CONFIG['duke']['topk']
            self.num  = CONFIG['duke']['num']
        elif 1400 - 10 < qN < 1400 + 10:
            self.topk = CONFIG['cuhk03']['topk']
            self.num  = CONFIG['cuhk03']['num']
        else:
            self.topk = CONFIG['recomm']['topk']
            self.num  = CONFIG['recomm']['num']

    def expand_sim(self, mat):
        N = mat.size(0)
        index = torch.arange(N, device=mat.device).repeat(2, 1)
        mat.scatter_(0, index, 0)
        # If not specific topk, each row is sum normalized: x_i = x_i / SUM(X).
        # Else softmax normalized: x_i = softmax(X).
        if self.topk is None or self.topk < 1:
            try:
                mat = mat / mat.sum(1).unsqueeze(1) * (1 - self.alpha)
            except:
                torch.cuda.empty_cache()
                sums = mat.sum(1)
                for x,y in zip(mat, sums):
                    x.div_(y).mul_(1 - self.alpha)
            mat.scatter_(0, index, self.alpha)
        else:
            self.smooth_topk_sim(mat, self.topk)
            mat.mul_(1 - self.alpha)
            mat.scatter_(0, index, self.alpha)
        torch.cuda.empty_cache()
        return True

    def smooth_topk_sim(self, mat, topk=10):
        topks, indices = mat.topk(topk, dim=1)
        # If not specific tau, just sum normalized the topk entries.
        if self.tau is None:
            normed_topks = topks / topks.sum(1).unsqueeze(1)
        else:
            topks.div_(self.tau)
            normed_topks = torch.softmax(topks, dim=1)
        mat.fill_(0)
        mat.scatter_(1, indices, normed_topks)
        return True

    def pre_update_gallery(self, g_feat):
        print("IIA Pre-update gallery ... ", end='')
        tic = time.time()
        for num in range(self.num):
            sims = torch.matmul(g_feat, g_feat.t())
            self.expand_sim(sims)
            g_feat = F.normalize(torch.mm(sims, g_feat))
        toc = time.time()
        print('Using {:.2f}s'.format(toc - tic))
        return g_feat

    def offline_update(self, q_feat, g_feat, q_id=None, g_id=None, q_cam=None, g_cam=None):
        g_feat = self.pre_update_gallery(g_feat)
        tic = time.time()
        for num in range(self.num):
            sims = torch.matmul(q_feat, g_feat.t())
            self.smooth_topk_sim(sims) 
            sims.mul_(1 - self.alpha)
            q_feat = q_feat * self.alpha + torch.matmul(sims, g_feat)
            q_feat = F.normalize(q_feat)
            if self.verbose and q_id is not None:
                dist = torch.matmul(q_feat, g_feat.t()).mul_(-1).add_(1)
                print("<alpha={:.2f}; topk={:d}; num={:d}>:  ".format(
                    self.alpha, self.topk, num + 1), end='')
                mAP, cmc = compute_score(dist, q_id, q_cam, g_id, g_cam, False)
                print_scores(mAP, cmc)
        dist = torch.matmul(q_feat, g_feat.t()).mul_(-1).add_(1)
        dist = dist.cpu().numpy()
        toc = time.time()
        print('IIA Offline update, Using {:.2f}s'.format(toc - tic))
        return dist

    def online_update(self, q_feat, g_feat, q_id=None, g_id=None, q_cam=None, g_cam=None):
        q_num = q_feat.size(0)
        feats = torch.cat((q_feat, g_feat))
        tic = time.time()
        for num in range(self.num):
            sims = torch.matmul(feats, feats.t())
            self.expand_sim(sims)
            feats = torch.mm(sims, feats)
            feats = F.normalize(feats)
            if self.verbose and q_id is not None:
                dist = torch.matmul(feats[:q_num], feats[q_num:].t()).mul_(-1).add(1)
                print("<alpha={:.2f}; topk={:d}; num={:d}>:  ".format(
                    self.alpha, self.topk, num + 1), end='')
                mAP, cmc = compute_score(dist, q_id, q_cam, g_id, g_cam, False)
                print_scores(mAP, cmc)
        dist = torch.matmul(feats[:q_num], feats[q_num:].t()).mul_(-1).add(1)
        dist = dist.cpu().numpy()
        toc = time.time()
        print('IIA Online update, Using {:.2f}s'.format(toc - tic))
        return dist

    def fit(self, q_feat, g_feat, q_id=None, g_id=None, q_cam=None, g_cam=None):
        q_feat = numpy2tensor(q_feat).to(self.device)
        g_feat = numpy2tensor(g_feat).to(self.device)
        q_feat = F.normalize(q_feat)
        g_feat = F.normalize(g_feat)
        if self.auto_infer_param:
            self.infer_param(q_feat.shape[0])
        if self.online:
            dist = self.online_update(q_feat, g_feat, q_id, g_id, q_cam, g_cam)
        else:
            dist = self.offline_update(q_feat, g_feat, q_id, g_id, q_cam, g_cam)
        return dist

    def __repr__(self):
        format_string  = self.__class__.__name__ + '('
        format_string += 'data_name={}, '.format(self.data_name)
        format_string += 'alpha={:.3f}, topk={:d}, '.format(self.alpha, self.topk)
        format_string += 'num={:d}, tau={:.3f}, '.format(self.num, self.tau)
        format_string += 'online={}, device={}, '.format(self.online, self.device)
        format_string += 'auto_infer_param={})'.format(self.auto_infer_param)
        return format_string

class AQE(object):
    def __init__(self, k=5):
        self.k = 5

    def fit(self, query, gallery, q2g=None, g2g=None):
        if q2g is None:
            q2g = torch.matmul(query, gallery.t())
        q2g_ind = q2g.topk(self.k)[1]
        q_exp = gallery[q2g_ind].mean(dim=1)
        new_q = torch.cat([query, q_exp], dim=1)

        if g2g is None:
            g2g = torch.matmul(gallery, gallery.t())
        ind = torch.arange(g2g.size(0), device=g2g.device)
        g2g.scatter_(1, ind.unsqueeze(1), 0)
        g2g_ind = g2g.topk(self.k)[1]
        g_exp = gallery[g2g_ind].mean(dim=1)
        new_g = torch.cat([gallery, g_exp], dim=1)
        dist = pairwise_distance(new_q, new_g)
        dist = dist.cpu().numpy()
        return dist
