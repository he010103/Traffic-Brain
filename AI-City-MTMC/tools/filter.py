import os
from os.path import join as opj
import cv2
import copy
import numpy as np

def filter_roi(roi, rects, width, height):
    new_rects = copy.deepcopy(rects)
    fil_rects = list()
    for fid_rect in rects:
        rect = fid_rect[1:5]
        if np.where(roi[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])] == 0)[0].shape[0] == 0:
            if (rect[3] - rect[1]) * (rect[2] - rect[0]) > 4900 and width - rect[2] > 10 and height - rect[3] > 10 and rect[0] > 10 and rect[1] > 10:
                fil_rects.append(fid_rect)
    return fil_rects

def parse_pt(pt_file):
    with open(pt_file) as f:
        lines = f.readlines()
    tid_rects = dict()
    for line in lines:
        line = line.strip().split(',')
        fid, tid = int(line[0]), int(line[1])
        rect = map(lambda x:float(x), line[2:6])
        rect[2] += rect[0]
        rect[3] += rect[1]
        if tid not in tid_rects:
            tid_rects[tid] = list()
        rect.insert(0, fid)
        tid_rects[tid].append(rect)
    return tid_rects

if __name__ == '__main__':
    data_dir = '../data/MOT/'
    save_dir = '../data/Filter_MOT/'
    img_dir = '../data/Track1/test/'
    for s in os.listdir(img_dir):
        for c in os.listdir(opj(img_dir, s)):
            print('processing cid: {} tid: {}'.format(c, s))
            roi = cv2.imread(opj(img_dir, s, c, 'roi.jpg'), 0)
            cv_img = cv2.imread(opj(img_dir, s, c, 'images/1.jpg'))
            height, width, _ = cv_img.shape
            tid_rects = parse_pt(opj(data_dir, '{}_{}.txt'.format(s, c)))
            filter_rects = dict()
            for tid in tid_rects:
                fid_rects = filter_roi(roi, tid_rects[tid], width, height)
                filter_rects[tid] = fid_rects
            new_tid_rects = {k:v for k, v in filter_rects.items() if len(v) > 2}
            new_fid_rects = dict()
            for tid in new_tid_rects:
                fid_rects = new_tid_rects[tid]
                for fid_rect in fid_rects:
                    fid, rect = fid_rect[0], fid_rect[1:5]
                    rect[2] -= rect[0]
                    rect[3] -= rect[1]
                    if fid not in new_fid_rects:
                        new_fid_rects[fid] = list()
                    new_fid_rects[fid].append([tid] + rect)
            f_w = open(opj(save_dir, '{}_{}.txt'.format(s, c)), 'wb')
            for fid in new_fid_rects:
                tid_rects = new_fid_rects[fid]
                for tid_rect in tid_rects:
                    tid, rect = tid_rect[0], tid_rect[1:5]
                    f_w.write(str(fid) + ',' + str(tid) + ',' + ','.join(map(str, rect)) + ',-1,-1,-1,-1\n')
            f_w.close()