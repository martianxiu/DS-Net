import os
from os.path import join, exists, isfile 
import json
import h5py

import numpy as np
import pickle 
import SharedArray as SA
from torch.utils.data import Dataset
import torch

from util.data_util import sa_create
from util.data_util import data_prepare

class DSNetDataset(Dataset):
    def __init__(self, transform, with_intensity=False, split='train', config=None, return_class=False):
        self.split = split
        self.with_intensity = with_intensity
        self.transform = transform
        self.config = config
        self.return_class = return_class
        # read file list
        if self.split == 'train':
            file_list = load_txt('dataset/train.txt')
        elif self.split == 'val':
            file_list = load_txt('dataset/val.txt')
        elif self.split == 'trainval':
            file_list = load_txt('dataset/train.txt') + load_txt('dataset/val.txt')
        elif self.split == 'test':
            file_list = load_txt('dataset/test.txt')
        else:
            raise NotImplementedError
            
        self.data, self.labels, self.dmgs = self.get_data_label(file_list)
        print('\nDataset preparation finished. ')
        print(f'  use intensity ? : {self.with_intensity}')       
        print('  length of {} data: {}\n'.format(split, self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        dmg = self.dmgs[idx]

        data, offset = pc_centralize(data)
        data, _ = pc_normalize(data)

        coord, feat = data[:, :3], data[:, 3:] 
        
        # augmentation
        if self.transform is not None:
            coord, feat, label = self.transform(coord, feat, label)

        if not self.with_intensity:
            feat = feat[:, :0]

        if self.return_class:
            return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label), torch.IntTensor([dmg]) 
        else:
            return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label)

    def get_class_weight(self):
        labels = np.concatenate(self.labels)
        u, c = np.unique(labels, return_counts=True)
        assert u.shape[0] == 3 

        total = np.sum(c)
        ratio = c / total
        
        weight = 1 / (ratio + 0.2)
        for i in range(weight.shape[0]):
            print(f'weight for class {i}: {weight[i]}')
        return weight.astype(np.float32)

    def get_data_label(self, file_list):
        f = h5py.File('dataset/pointcloud.h5', 'r')
        data = []
        labels = []
        dmgs = []
        for uid in file_list:
            data.append(
                np.concatenate(
                    (f[uid]['points'][:], f[uid]['intensity'][:].reshape(-1,1)),
                    1
                ) 
            )
            labels.append(f[uid]['per_point_labels'][:])
            dmgs.append(f[uid]['damage'][()])
        return data, labels, dmgs 

# utility functions
def load_txt(txt):
    with open(txt, mode='r') as f:
        files = f.readlines()
        files = [x.split('\n')[0] for x in files]
        f.close()
    return files

def pc_translation(pc, translate_range=0.2):
    delta_xyz = np.random.uniform(low=-translate_range, high=translate_range, size=[3])
    pc[:, :3] = pc[:, :3] + delta_xyz
    print(delta_xyz)
    return pc

def pc_rotation_z(pc, rotation_angle=None):
    if pc.shape[1] > 3:
        xyz, features = pc[:, :3], pc[:, 3:]
    else:
        xyz, features = pc, None
    rotated_data = np.zeros(pc.shape, dtype=np.float32)
    if rotation_angle == None:
        rotation_angle = np.random.uniform() * 2 * np.pi
    else:
        rotation_angle = rotation_angle
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]])
    rotated_xyz = np.dot(xyz, rotation_matrix)
    if features is not None:
        rotated_data = np.concatenate((rotated_xyz, features), -1)
    else:
        rotated_data = rotated_xyz
    return rotated_data, rotation_angle

def pc_sample(pc, N):
    M = pc.shape[0]
    fake_id = np.arange(M)
    if M >= N:
        ind = np.random.choice(fake_id, N, replace=False)
    else:
        ind = np.random.choice(fake_id, N, replace=True)
    return pc[ind]

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_centralize(pc, offset=None):
    if offset is None:
        offset = np.mean(pc[:, :3], axis=0)
        pc[:, :3] = pc[:, :3] - offset
    else:
        pc[:, :3] = pc[:, :3] - offset
    return pc, offset

def pc_normalize(pc, m=None):
    if m is None:
        m = np.max(np.sqrt(np.sum(pc[:, :3]**2, axis=1)))
        pc[:, :3] = pc[:, :3] / m
    else:
        pc[:, :3] = pc[:, :3] / m
    return pc, m

def jitter_pointcloud(pointcloud, mean=0, sigma=0.01, clip=0.05):
    # jitter = np.clip(sigma * np.random.randn(pointcloud.shape[0], 3), -1 * clip, clip)
    jitter = np.clip(np.random.normal(mean, sigma, (pointcloud.shape[0], 3)), -1 * clip, clip)
    pointcloud[:, :3] += jitter
    return pointcloud

def scale_pointcloud(pc, anisotropic=False):
    if anisotropic:
        size = 3
    else:
        size= 1
    scale = np.random.uniform(low=2./3., high=3./2., size=[size])
    pc[:, :3] *= scale
    return pc

def shift_pointcloud(pc):
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    pc[:, :3] += shift
    return pc
    
def scale_and_shift(pc):
    scale = np.random.uniform(low=2./3., high=3./2., size=[3])
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    pc[:, :3] *= scale
    pc[:, :3] += shift
    return pc 

def scale_and_shift_vote(pc):
    # scale = np.random.uniform(low=2./3., high=3./2., size=[3])
    # shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    scale = np.random.uniform(low=0.8, high=1.2, size=[3])
    shift = np.random.uniform(low=-0.1, high=0.1, size=[3])
    # scale = np.random.uniform(low=0.9, high=1.1, size=[3])
    # shift = np.random.uniform(low=-0.00, high=0.00, size=[3])
    # scale = np.random.uniform(low=0.8, high=1.2, size=[3])
    # shift = np.random.uniform(low=-0.00, high=0.00, size=[3])
    # scale = np.random.uniform(low=1.0, high=1.0, size=[3])
    # shift = np.random.uniform(low=-0.00, high=0.00, size=[3])
    pc[:, :3] *= scale
    pc[:, :3] += shift
    return pc 

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc
