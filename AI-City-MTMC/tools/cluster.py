import os
from os.path import join as opj
import scipy.io as scio
import cv2
import numpy as np
import pickle
from scipy import spatial
import copy
import multiprocessing
from math import *
from sklearn import preprocessing

def compute_dis(Lat_A, Lng_A, Lat_B, Lng_B):
    ra = 6378.140  
    rb = 6356.755  
    flatten = (ra - rb) / ra  
    rad_lat_A = radians(Lat_A)
    rad_lng_A = radians(Lng_A)
    rad_lat_B = radians(Lat_B)
    rad_lng_B = radians(Lng_B)
    pA = atan(rb / ra * tan(rad_lat_A))
    pB = atan(rb / ra * tan(rad_lat_B))
    cos_num = sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B)
    if cos_num > 1.: cos_num = 1.
    xx = acos(cos_num)
    if cos(xx / 2.0) == 0 or sin(xx / 2.) == 0: return 0.
    c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / (cos(xx / 2.) ** 2)
    c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / (sin(xx / 2.) ** 2)
    dr = flatten / 8. * (c1 - c2)
    distance = 1000*ra * (xx + dr)
    return distance

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

def project(H, point):
    res = np.dot(np.mat(H).I, point)
    return [float(res[0]/res[-1]), float(res[1]/res[-1])]

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

def undist(x, y, cid):
    if cid != 5 and cid != 35: return x, y
    if cid == 5:
        camera_matrix = np.array([[1280.000000000000000, 0.000000000000000, 640.000000000000000], [0.000000000000000, 1280.000000000000000, 480.000000000000000], [0.000000000000000, 0.000000000000000, 1.000000000000000]], dtype=np.float32)
        dist_coeffs = np.array([-0.600000023841858, 0.000000000000000, 0.000000000000000, 0.000000000000000], dtype=np.float32)
    elif cid == 35:
        camera_matrix = np.array([[1920.000000000000000, 0.000000000000000, 960.000000000000000], [0.000000000000000, 1920.000000000000000, 540.000000000000000], [0.000000000000000, 0.000000000000000, 1.000000000000000]], dtype=np.float32)
        dist_coeffs = np.array([ -0.600000023841858, 0.000000000000000, 0.000000000000000, 0.000000000000000], dtype=np.float32)
    points = np.array([[[x, y]]], dtype=np.float64)
    pts = cv2.undistortPoints(points, camera_matrix, dist_coeffs, None, camera_matrix)
    new_x, new_y = pts[0][0]
    return new_x, new_x

def get_time_gis(data_dir, roi_dir, cid_bias, cid_arr):
    txt_paths = os.listdir(data_dir)
    txt_paths = sorted(txt_paths, key=lambda x: int(x.split('.')[0][-3:]))
    cid_tid_time_gis = dict()
    for txt_path in txt_paths:
        cid = int(txt_path.split('.')[0][-3:])
        if cid not in cid_tid_time_gis: cid_tid_time_gis[cid] = dict()
        cur_bias = cid_bias[cid]
        cur_arr = cid_arr[cid]
        roi = cv2.imread(opj(roi_dir, '{}.jpg'.format(txt_path.split('.')[0])), 0)
        img_rects = parse_pt(opj(data_dir, txt_path))
        for fid in img_rects:
            tid_rects = img_rects[fid]
            time = cur_bias + fid / 10. if cid != 15 else cur_bias + fid / 8. 
            for tid_rect in tid_rects:
                tid = tid_rect[0]
                if tid not in cid_tid_time_gis[cid]: cid_tid_time_gis[cid][tid]= list()
                rect = tid_rect[1:5]
                cx = 0.5*rect[0] + 0.5*rect[2]
                cy = 0.5*rect[1] + 0.5*rect[3]
                px = cx
                py = rect[3]
                px, py = undist(px, py, cid)
                gis_lat, gis_lng = project(cur_arr, np.array([[px], [py], [1]]))
                cid_tid_time_gis[cid][tid].append([time, gis_lat, gis_lng])
    return cid_tid_time_gis

def normalize(nparray, order=2, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray

def write_sim_matrix(cid_tids, score_thr, save_name):
    count = len(cid_tids)
    print('count: ', count)
    pool = multiprocessing.Pool(processes=96)
    print('iterating the (i, j)')
    params_list = [(i, j, cid_tids[i], cid_tids[j]) for i in range(count) for j in range(count) if i < j and cid_tids[i][0] != cid_tids[j][0]] 
    res_list = pool.map(compute_sim, params_list)
    q_arr = np.array([cid_tid_fea[cid_tids[i]] for i in range(count)])
    g_arr = np.array([cid_tid_fea[cid_tids[i]] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)
    sim_matrix = np.matmul(q_arr, g_arr.T)
    
    for idx, res in enumerate(res_list):
        i, j, cur_sim = res
        sim_matrix[i, j] = cur_sim * sim_matrix[i, j]
        sim_matrix[j, i] = sim_matrix[i][j]

    print(sim_matrix)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][0] == cid_tids[j][0]:
                sim_matrix[i, j] = 0.
    sim_matrix[sim_matrix < score_thr] = 0
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def write_match_martix(sim_matrix, cid_tids, score_thr):
    print('writing match martix')
    count = len(cid_tids)
    match_matrix = np.zeros((count, count))
    cluster_dict = {k:[k] for k in range(count)}
    sim_matrix_cpy = sim_matrix.copy()
    while np.max(sim_matrix_cpy) > 0.:
        max_sim = np.max(sim_matrix_cpy)
        idx1, idx2 = np.where(sim_matrix_cpy == max_sim)[0][0], np.where(sim_matrix_cpy == max_sim)[1][0]
        sim_matrix_cpy[idx1, idx2] = 0.
        sim_matrix_cpy[idx2, idx1] = 0.
        all_idxs1 = cluster_dict[idx1]
        all_cids1 = list(set([cid_tids[idx][0] for idx in all_idxs1]))
        all_idxs2 = cluster_dict[idx2]
        all_cids2 = list(set([cid_tids[idx][0] for idx in all_idxs2]))
        ins = list(set(all_cids1).intersection(set(all_cids2)))
        if len(ins) == 0:
            all_idxs = all_idxs1 + all_idxs2

            match_matrix[idx1][idx2] = 1
            match_matrix[idx2][idx1] = 1

            for i1 in all_idxs1: 
                for i2 in all_idxs2: 
                    sim_matrix_cpy[i1, i2] = 0
                    sim_matrix_cpy[i2, i1] = 0
            cur_cluster = cluster_dict[idx1] + cluster_dict[idx2]
            for idx in all_idxs:
                cluster_dict[idx] = cur_cluster
    return match_matrix, cluster_dict

def get_match(cluster_dict):
    cluster = list()
    visited = list()
    for idx in cluster_dict:
        if idx not in visited:
            visited += cluster_dict[idx]
            cluster.append(cluster_dict[idx])
    return cluster

def compute_sim(params):
    i, j, cid_tid1, cid_tid2 = params
    track1 = cid_tid_time_gis[cid_tid1[0]][cid_tid1[1]]
    track2 = cid_tid_time_gis[cid_tid2[0]][cid_tid2[1]]
    return [i, j, track_sim(track1, track2)]

def get_labels(cid_tids, score_thr=0.65, save_name='model.data'):
    sim_matrix = write_sim_matrix(cid_tids, score_thr, save_name)
    match_matrix, cluster_dict = write_match_martix(sim_matrix, cid_tids, score_thr)
    labels = get_match(cluster_dict)
    return labels

# spatial and temportal information
def track_sim(track1, track2):
    track1 = sorted(track1, key=lambda x: x[0])
    track2 = sorted(track2, key=lambda x: x[0])
    e1, l1 = track1[0][0], track1[-1][0]
    e2, l2 = track2[0][0], track2[-1][0]
    i1, i2 = max(e1, e2), min(l1, l2)
    if i1 < i2: 
        cut_i1 = i1 + 0.1*(i2 - i1)
        cut_i2 = i1 + 0.9*(i2 - i1)
        f_track1 = filter(lambda x: x[0] >= cut_i1 and x[0] <= cut_i2, track1)
        f_track2 = filter(lambda x: x[0] >= cut_i1 and x[0] <= cut_i2, track2)

        dis_sim = 1.
        dis_arr =  list()
        for t1 in f_track1:
            time1, lat1, lng1 = t1[0], t1[1], t1[2]
            t2 = filter(lambda x: x[0] > time1 - 0.05 and x[0] < time1 + 0.05, track2)
            if len(t2) < 1: continue
            t2 = t2[0]
            time2, lat2, lng2 = t2[0], t2[1], t2[2]
            dis_arr.append(compute_dis(lat1, lng1, lat2, lng2))
        if len(dis_arr) > 0:
            mean_dis = np.mean(np.array(dis_arr)) 
            if mean_dis > 300: return 0.
            dis_sim = max(0, 1. - pow((mean_dis/200.),2))
            return dis_sim

        return 1.

    elif i1 > i2:
        speed_sim = 1.
        if l1 < e2 + 1: 
            lat1, lng1 = track1[-1][1], track1[-1][2]
            lat2, lng2 = track2[0][1], track2[0][2]
            dis = compute_dis(lat1, lng1, lat2, lng2)
            speed = dis / (e2 - l1)
            if speed > 30: return speed_sim
            speed_sim = max(0, 1. - pow((speed - 10.)/30., 2))
            return speed_sim
            
        elif l2 < e1 + 1:
            speed_sim = 1.
            lat1, lng1 = track1[0][1], track1[0][2]
            lat2, lng2 = track2[-1][1], track2[-1][2]
            dis = compute_dis(lat1, lng1, lat2, lng2)
            speed = dis / (e1 - l2)
            if speed > 30: return speed_sim
            speed_sim = max(0, 1. - pow((speed - 10.)/30., 2))
            return speed_sim
    return 1.

if __name__ == '__main__':
    data_dir = '/mnt/lustre/share/hezhiqun/AICity/Track1/MOT/test/filter_mot3/'
    scene_name = ['S02', 'S05']
    cid_bias = parse_bias('/mnt/lustre/share/hezhiqun/AICity/Track1/cam_timestamp', scene_name)
    cid_arr = homography('../data/Track1/calibration/')
    roi_dir = '../data/roi/'
    global cid_tid_time_gis
    cid_tid_time_gis = get_time_gis(data_dir, roi_dir, cid_bias, cid_arr)
    all_cids = sorted(cid_bias.keys())

    fea_dir = '../data/trajectory/'
    global cid_tid_fea
    cid_tid_fea = dict()
    scene_cluster = [[6,7,8,9], [10,16,17,18,19,20,21,22,23,24,25,26,27,28,29,33,34,35,36]]

    for txt_path in os.listdir(fea_dir):
        cid = int(txt_path.split('.')[0][-3:])
        with open(opj(fea_dir, txt_path)) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if len(line) == 2: continue
            tid = int(line[1])
            enter_time = float(line[2])
            leave_time = float(line[3])
            fea = map(float, line[4:])
            if (cid, tid) not in cid_tid_fea:
                cid_tid_fea[(cid, tid)] = fea

    cid_tids1 = sorted([key for key in cid_tid_fea.keys() if key[0] in scene_cluster[0]])
    clu1 = get_labels(cid_tids1, save_name='1')
    cid_tids2 = sorted([key for key in cid_tid_fea.keys() if key[0] in scene_cluster[1]], key=lambda x: x[0])
    clu2 = get_labels(cid_tids2, save_name='2')

    new_clu1 = list()
    for c_list in clu1:
        if len(c_list) <= 1: continue
        new_clu1.append([cid_tids1[c] for c in c_list])
    print('new_clu1: ', len(new_clu1))

    
    new_clu2 = list()
    for c_list in clu2:
        if len(c_list) <= 1: continue
        new_clu2.append([cid_tids2[c] for c in c_list])
    print('new_clu2: ', len(new_clu2))

    all_clu = new_clu1 + new_clu2 

    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    pickle.dump({'cluster': cid_tid_label}, open('test_cluster.data', 'wb'))