# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, camids, tids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    tids = torch.tensor(tids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, tids

def val_collate_fn(batch):
    imgs, pids, camids, tids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    tids = torch.tensor(tids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, tids
