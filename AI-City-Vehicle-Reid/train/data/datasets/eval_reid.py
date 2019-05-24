# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import pytrec_eval

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=100):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    # mAP = np.mean(all_AP)
    mAP = pytrec_mAP(distmat, q_pids, g_pids, q_camids, g_camids, topk=max_rank)

    return all_cmc, mAP

def pytrec_mAP(distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None, topk=100):
    m, n = distmat.shape
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    gt = dict()
    pt = dict()
    for i in range(m):
        gt[str(i)] = dict()
        pt[str(i)] = dict()
        valid = ((gallery_ids != query_ids[i]) |
                 (gallery_cams != query_cams[i]))
        valid_mat = distmat[i, valid]
        match = (gallery_ids[valid] == query_ids[i])
        indices = np.argsort(valid_mat)
        for j in range(valid_mat.shape[0]):
            if match[j]: gt[str(i)][str(j)] = 1
            else: gt[str(i)][str(j)] = 0
        for j, idx in enumerate(indices):
            if j < 100:
                pt[str(i)][str(idx)] = 100 - j
    evaluator = pytrec_eval.RelevanceEvaluator(gt, {'map_cut'})
    res = evaluator.evaluate(pt)
    map_list = [v['map_cut_{}'.format(topk)] for k, v in res.items()]
    return np.mean(map_list)