# srun --mpi=pmi2 -p VI_UC_TITANXP -n1 --gres=gpu:4 python test.py

import os
from os.path import join as opj
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import sys   
import re
from sklearn import preprocessing
import multiprocessing

sys.path.append('../train/')
from modeling.baseline import Baseline
from metrics import track_reranking_weight_feat
import pickle

def list_pictures(directory):
    imgs = sorted([opj(directory, img) for img in os.listdir(directory)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return imgs

if __name__ == '__main__':
    # get track information
    test_dir = '/mnt/lustre/share/hezhiqun/AICity/Track2/AIC_Reid/submit_test/'
    test_img_paths = [path for path in list_pictures(test_dir)]
    print('gallery img num: ', len(test_img_paths))
    track_file = '/mnt/lustre/share/hezhiqun/AICity/Track2/test_track.txt'    
    tids = -1*np.ones(len(test_img_paths))
    with open(track_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            imgs = line.strip().split(' ')
            for img in imgs:
                index = int(img.split('.')[0]) - 1
                tids[index] = i
    unique_tids = sorted(np.asarray(list(set(tids))))
    print('tids num: ', len(tids))

    print('loading pickle...')


    qf = pickle.load(open('data/submit_res101_qf_Track1_256.data', 'rb'))['qf']
    gf = pickle.load(open('data/submit_res101_gf_Track1_256.data', 'rb'))['gf']

    qf2 = pickle.load(open('data/submit_res50_qf_Track1.data', 'rb'))['qf']
    gf2 = pickle.load(open('data/submit_res50_gf_Track1.data', 'rb'))['gf']

    qf3 = pickle.load(open('data/submit_res50_qf_Track2.data', 'rb'))['qf']
    gf3 = pickle.load(open('data/submit_res50_gf_Track2.data', 'rb'))['gf']

    qf4 = pickle.load(open('data/submit_res101_qf_Track2_256.data', 'rb'))['qf']
    gf4 = pickle.load(open('data/submit_res101_gf_Track2_256.data', 'rb'))['gf']

    qf5 = pickle.load(open('data/submit_se_res101_qf_Track1.data', 'rb'))['qf']
    gf5 = pickle.load(open('data/submit_se_res101_gf_Track1.data', 'rb'))['gf']

    qf6 = pickle.load(open('data/submit_res50_qf_Track1_guangzhou.data', 'rb'))['qf']
    gf6 = pickle.load(open('data/submit_res50_gf_Track1_guangzhou.data', 'rb'))['gf']

    qf7 = pickle.load(open('data/submit_res50_qf_Track1_shenzhen.data', 'rb'))['qf']
    gf7 = pickle.load(open('data/submit_res50_gf_Track1_shenzhen.data', 'rb'))['gf']

    qf8 = pickle.load(open('data/submit_res101_qf_Track2_new.data', 'rb'))['qf']
    gf8 = pickle.load(open('data/submit_res101_gf_Track2_new.data', 'rb'))['gf']

    qf9 = pickle.load(open('data/submit_res50_qf_Track2_new.data', 'rb'))['qf']
    gf9 = pickle.load(open('data/submit_res50_gf_Track2_new.data', 'rb'))['gf']

    qf = preprocessing.normalize(qf, norm='l2')
    gf = preprocessing.normalize(gf, norm='l2')
    qf2 = preprocessing.normalize(qf2, norm='l2')
    gf2 = preprocessing.normalize(gf2, norm='l2')
    qf3 = preprocessing.normalize(qf3, norm='l2')
    gf3 = preprocessing.normalize(gf3, norm='l2')
    qf4 = preprocessing.normalize(qf4, norm='l2')
    gf4 = preprocessing.normalize(gf4, norm='l2')
    qf5 = preprocessing.normalize(qf5, norm='l2')
    gf5 = preprocessing.normalize(gf5, norm='l2')
    qf6 = preprocessing.normalize(qf6, norm='l2')
    gf6 = preprocessing.normalize(gf6, norm='l2')
    qf7 = preprocessing.normalize(qf7, norm='l2')
    gf7 = preprocessing.normalize(gf7, norm='l2')
    qf8 = preprocessing.normalize(qf8, norm='l2')
    gf8 = preprocessing.normalize(gf8, norm='l2')
    qf9 = preprocessing.normalize(qf9, norm='l2')
    gf9 = preprocessing.normalize(gf9, norm='l2')

    print('reranking...')

    print('track_ranking...')
    track_dist1 = track_reranking_weight_feat(qf, gf, tids, unique_tids)
    track_dist2 = track_reranking_weight_feat(qf2, gf2, tids, unique_tids)
    track_dist3 = track_reranking_weight_feat(qf3, gf3, tids, unique_tids)
    track_dist4 = track_reranking_weight_feat(qf4, gf4, tids, unique_tids)
    track_dist5 = track_reranking_weight_feat(qf5, gf5, tids, unique_tids)
    track_dist6 = track_reranking_weight_feat(qf6, gf6, tids, unique_tids)
    track_dist7 = track_reranking_weight_feat(qf7, gf7, tids, unique_tids)
    track_dist8 = track_reranking_weight_feat(qf8, gf8, tids, unique_tids)
    track_dist9 = track_reranking_weight_feat(qf9, gf9, tids, unique_tids)
    track_dist = track_dist1 * track_dist2 * track_dist3 * track_dist4 * track_dist5 * track_dist6 * track_dist7 * track_dist8 * track_dist9 * track_dist9

    print('saving...')
    indices = np.argsort(track_dist, axis=1)[:, :100]
    m, n = indices.shape
    print('m: {}  n: {}'.format(m, n))
    with open('track2.txt', 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())
    print(indices[0])
    print(indices.shape)