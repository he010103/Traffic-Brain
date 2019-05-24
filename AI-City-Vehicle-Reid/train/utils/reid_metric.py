# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
from scipy.spatial.distance import cdist

import torch
from ignite.metrics import Metric


from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking, re_ranking_numpy


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []

    def update(self, output):
        feat, pid, camid, tid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.tids.extend(np.asarray(tid))
        self.unique_tids = list(set(self.tids))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        gallery_tids = np.asarray(self.tids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]

        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)

        return cmc, mAP

class Track_R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(Track_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []

    def update(self, output):
        feat, pid, camid, tid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.tids.extend(np.asarray(tid))
        self.unique_tids = list(set(self.tids))

    def track_ranking(self, qf, gf, gallery_tids, unique_tids):
        origin_dist = cdist(qf, gf)
        m, n = qf.shape[0], gf.shape[0]
        feature_dim = qf.shape[1]
        gallery_tids = np.asarray(gallery_tids)
        unique_tids = np.asarray(unique_tids)
        track_gf = np.zeros((len(unique_tids), feature_dim))
        dist = np.zeros((m, n))
        gf_tids = sorted(list(set(gallery_tids)))
        for i, tid in enumerate(gf_tids):
            track_gf[i, :] = np.mean(gf[gallery_tids == tid, :] , axis=0)
        # track_dist = cdist(qf, track_gf)
        track_dist = re_ranking_numpy(qf, track_gf, k1=8, k2=3, lambda_value=0.3)
        # track_dist = re_ranking_numpy(qf, track_gf, k1=5, k2=3, lambda_value=0.3)
        for i, tid in enumerate(gf_tids):
            dist[:, gallery_tids == tid] = track_dist[:, i:(i+1)]
        for i in range(m):
            for tid in gf_tids:
                min_value = np.min(origin_dist[i][gallery_tids == tid])
                min_index = np.where(origin_dist[i] == min_value)
                min_value = dist[i][min_index[0][0]]
                dist[i][gallery_tids == tid] = min_value + 0.000001
                dist[i][min_index] = min_value
        return dist

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        gallery_tids = np.asarray(self.tids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]

        qf = qf.cpu().numpy()
        gf = gf.cpu().numpy()
        distmat = self.track_ranking(qf, gf, gallery_tids, self.unique_tids)
        
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.tids = []

    def update(self, output):
        feat, pid, camid, tid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.tids.extend(np.asarray(tid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=30, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)

        return cmc, mAP