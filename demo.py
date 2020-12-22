#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-12-22 22:09:43
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os, sys
import numpy as np
import time, argparse
import torch
from utils import *
from iia import IIA, AQE

def parse_args():
    parser = argparse.ArgumentParser(description='Post-Processing with IIA')
    parser.add_argument('--dname', dest='data_name', type=str,   default='')
    parser.add_argument('--fpath', dest='fpath',     type=str,   default='',
                        help='path for extracted features')
    parser.add_argument('--dev',   dest='device',    type=str,   default='0')
    parser.add_argument('--alpha', dest='alpha',     type=float, default=None)
    parser.add_argument('--tau',   dest='tau',      type=float, default=None)
    parser.add_argument('--topk',  dest='topk',      type=int,   default=None)
    parser.add_argument('--num',   dest='num',       type=int,   default=None)
    parser.add_argument('--off',   dest='offline',   action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.device.isdigit(): args.device = int(args.device)
    if not os.path.exists(args.fpath):
        raise IOError('Feature file={} is not exist'.format(args.fpath))
    q_feat, q_pid, q_cam, g_feat, g_pid, g_cam = load_data(args.fpath, dtype='torch')

    ori_dist = pairwise_distance(q_feat, g_feat, 'cosine').cpu().numpy()
    ori_ap, ori_cmc = compute_score(ori_dist, q_pid, q_cam, g_pid, g_cam, False)
    print('Score Original: ')
    print_scores(ori_ap, ori_cmc)

    iia_runner = IIA(data_name=args.data_name, alpha=args.alpha, 
                     topk=args.topk, num=args.num, tau=args.tau, 
                     device=args.device, online=not args.offline, verbose=False)
    print(iia_runner)
    iia_dist = iia_runner.fit(q_feat, g_feat, q_pid, g_pid, q_cam, g_cam)
    iia_ap, iia_cmc = compute_score(iia_dist, q_pid, q_cam, g_pid, g_cam, False)
    print('Score After IIA: ')
    print_scores(iia_ap, iia_cmc)
