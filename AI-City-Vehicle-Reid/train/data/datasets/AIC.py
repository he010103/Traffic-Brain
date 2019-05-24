# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class AIC(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 776 (+1 for background)
    # images: 37778 (train) + 1578 (query) + 11579 (gallery)
    """
    dataset_dir = 'AIC'

    def __init__(self, root='.', verbose=True, **kwargs):
        super(AIC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        

        self.train_dir = osp.join(self.dataset_dir, 'Track2_train_Track1_test')
        self.query_dir = osp.join(self.dataset_dir, 'hezhiqun_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'hezhiqun_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> AIC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_tids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_tids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_tids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d*)')

        pid_container = set()
        tid_container = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            tid_container.add((pid, camid))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        tid2label = {tid: label for label, tid in enumerate(tid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            
            tid = int(tid2label[(pid, camid)])
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: 
                pid = pid2label[pid]                
            dataset.append((img_path, pid, camid, tid))

        return dataset
