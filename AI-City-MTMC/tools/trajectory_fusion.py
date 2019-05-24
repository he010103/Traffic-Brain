import os
import sys
from os.path import join as opj
import base64
import struct
import copy
import numpy as np
from scipy.spatial.distance import cdist
import cv2

def parse_pt(pt_file):
    with open(pt_file) as f:
        lines = f.readlines()
    img_rects = dict()
    for line in lines:
        line = line.strip().split(',')
        fid, tid = int(float(line[0])), int(float(line[1]))
        rect = map(lambda x:int(float(x)), line[2:6])
        rect[2] += rect[0]
        rect[3] += rect[1]
        fea = base64.b64decode(line[-1])
        fea = struct.unpack('{}f'.format(len(fea)/4), fea)
        if tid not in img_rects:
            img_rects[tid] = list()
        rect.insert(0, fid)
        rect.append(fea)
        img_rects[tid].append(rect)
    return img_rects

def parse_bias(timestamp_dir, scene_name):
    cid_bias = dict()
    for sname in scene_name:
        with open(opj(timestamp_dir, sname + '.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                cid = int(line[0][2:])
                bias = float(line[1])
                if cid not in cid_bias: cid_bias[cid] = bias
    return cid_bias 

def homography(data_dir):
    cid_arr = dict()    
    for cid_name in os.listdir(data_dir):
        if '.' in cid_name: continue
        cid = int(cid_name[1:])
        cal_path = opj(data_dir, cid_name, 'calibration.txt')
        with open(cal_path) as f:
            lines = f.readlines()
        datas = lines[0].strip().split(':')[-1].strip().split(';')
        arr = list()
        for data in datas:
            arr.append(map(float, data.split(' ')))
        arr = np.array(arr)
        cid_arr[cid] = arr
    return cid_arr

if __name__ == '__main__':
    data_dir = '../data/feature/'
    roi_dir = '../data/roi/'
    save_dir = '../data/trajectory/'
    scene_name = ['S02', 'S05']


    cid_bias = parse_bias('../data/Track1/cam_timestamp', scene_name)
    cid_arr = homography('../data/Track1/calibration/')
    txt_paths = os.listdir(data_dir)
    txt_paths = filter(lambda x: '.txt' in x, sorted(txt_paths, key=lambda x: int(x.split('.')[0][-3:])))
    for txt_path in txt_paths:
        print('processing {}...'.format(txt_path))
        cid = int(txt_path.split('.')[0][-3:])
        f_w = open(opj(save_dir, txt_path), 'wb')
        cur_bias = cid_bias[cid]
        roi = cv2.imread(opj(roi_dir, '{}.jpg'.format(txt_path.split('.')[0])), 0)
        img_rects = parse_pt(opj(data_dir, txt_path))
        tid_data = dict()
        for tid in img_rects:
            rects = img_rects[tid]
            if len(rects) == 0: continue
            tid_data[tid] = [cid]
            rects = sorted(rects, key=lambda x: x[0])
            if cid != 15:
                tid_data[tid] += [cur_bias + rects[0][0] / 10., cur_bias + rects[-1][0] / 10.] # [enter, leave]
            else:
                tid_data[tid] += [cur_bias + rects[0][0] / 8., cur_bias + rects[-1][0] / 8.] # [enter, leave]

            all_fea = np.array([rect[-1] for rect in rects[int(0.3*len(rects)):int(0.7*len(rects)) + 1]])
            mean_fea = np.mean(all_fea, axis=0)

            tid_data[tid] += mean_fea.tolist()
        for tid in tid_data:
            data = tid_data[tid]
            data.insert(1, tid)
            f_w.write(' '.join(map(str, data)) + '\n')
        f_w.close()
