import os
from os.path import join as opj
from scipy import spatial
import copy
import numpy as np
import cv2
import pickle
from math import *
    
def parse_pt(pt_file):
    with open(pt_file) as f:
        lines = f.readlines()
    img_rects = dict()
    for line in lines:
        line = line.strip().split(',')
        fid, tid = int(line[0]), int(line[1])
        rect = map(lambda x:int(float(x)), line[2:6])
        rect[2] += rect[0]
        rect[3] += rect[1]
        if fid not in img_rects:
            img_rects[fid] = list()
        rect.insert(0, tid)
        img_rects[fid].append(rect)
    return img_rects

if __name__ == '__main__':
    data_dir = '../data/Filter_MOT/'
    roi_dir = '../data/roi/'
    scene_name = ['S02', 'S05']
    scene_cluster = [[6,7,8,9], [10,16,17,18,19,20,21,22,23,24,25,26,27,28,29,33,34,35,36]]

    map_tid = pickle.load(open('test_cluster.data', 'rb'))['cluster']

    f_w = open('track1.txt', 'wb')
    txt_paths = os.listdir(data_dir)
    txt_paths = sorted(txt_paths, key=lambda x: int(x.split('.')[0][-3:]))
    for txt_path in txt_paths:
        cid = int(txt_path.split('.')[0][-3:])
        
        roi = cv2.imread(opj(roi_dir, '{}.jpg'.format(txt_path.split('.')[0])), 0)
        height, width = roi.shape
        img_rects = parse_pt(opj(data_dir, txt_path))
        for fid in img_rects:
            tid_rects = img_rects[fid]
            for tid_rect in tid_rects:
                tid = tid_rect[0]
                rect = tid_rect[1:]
                cx = 0.5*rect[0] + 0.5*rect[2]
                cy = 0.5*rect[1] + 0.5*rect[3]
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                rect[2] -= rect[0]
                rect[3] -= rect[1]
                rect[0] = max(0, rect[0])
                rect[1] = max(0, rect[1])
                x1, y1 = max(0, cx - 0.5*w - 20), max(0, cy - 0.5*h - 20)
                x2, y2 = min(width-x1, w + 40), min(height-y1, h + 40)
                
                new_rect = map(int, [x1, y1, x2, y2])
                rect = map(int, rect)
                if (cid, tid) in map_tid:
                    new_tid = map_tid[(cid, tid)]
                    f_w.write(str(cid) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' + ' '.join(map(str, new_rect)) + ' -1 -1' '\n')
    f_w.close()