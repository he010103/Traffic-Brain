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

import torch
from torch.optim import lr_scheduler
from torch import nn
from torchvision import transforms
from torchvision.datasets.folder import default_loader

sys.path.append('../train/')
from modeling.baseline import Baseline
from metrics import track_reranking_weight_feat
import pickle

test_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature(model, inputs):
    with torch.no_grad():
        features = torch.FloatTensor()
        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        for i in range(2):
            if i == 1:
                # flip
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f
        features = torch.cat((features, ff), 0)
    return features.cpu().data.numpy()

def list_pictures(directory):
    imgs = sorted([opj(directory, img) for img in os.listdir(directory)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return imgs

def read_image(img_path):
    img = default_loader(img_path)
    img = test_transform(img)
    img = img.unsqueeze(0)
    return img

if __name__ == '__main__':
    model_path = 'your model path'
    model = Baseline(10000, 1, model_path, 'bnneck', 'before', 'resnet50', 'self')


    model = model.to('cuda')
    model = nn.DataParallel(model)
    
    resume = torch.load(model_path)
    model.load_state_dict(resume)
    
    model.eval()

    print('create multiprocessing...')
    pool = multiprocessing.Pool(processes=32)
    print('after create multiprocessing...')

    query_dir = './submit_query/'
    test_dir = './submit_test/'
    query_img_paths = [path for path in list_pictures(query_dir)]
    test_img_paths = [path for path in list_pictures(test_dir)]

    batch_size = 256
    
    qf = np.zeros((len(query_img_paths), 2048))
    
    for i in tqdm(range( int(np.ceil(len(query_img_paths)/batch_size)) )):
        cur_query_img = pool.map(read_image, query_img_paths[i*batch_size:(i+1)*batch_size])
        cur_query_img = torch.cat(cur_query_img, 0)
        if len(cur_query_img) == 0: break
        cur_qf = extract_feature(model, cur_query_img)
        qf[i*batch_size:(i+1)*batch_size, :] = cur_qf

    gf = np.zeros((len(test_img_paths), 2048))
    
    for i in tqdm(range( int(np.ceil(len(test_img_paths)/batch_size)) )):
        cur_test_img = pool.map(read_image, test_img_paths[i*batch_size:(i+1)*batch_size])
        cur_test_img = torch.cat(cur_test_img, 0)
        if len(cur_test_img) == 0: break
        cur_gf = extract_feature(model, cur_test_img)
        gf[i*batch_size:(i+1)*batch_size, :] = cur_gf

    pickle.dump({'qf': qf}, open('submit_res50_qf.data', 'wb'))
    pickle.dump({'gf': gf}, open('submit_res50_gf.data', 'wb'))