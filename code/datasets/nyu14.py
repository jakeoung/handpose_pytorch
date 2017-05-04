"""
Some parts of the codes are based on
[1] https://github.com/moberweger/deep-prior/blob/master/src/data/importers.py
"""

import os
import numpy as np
import torch
import torch.utils.data as data
import scipy.io
from .GetH5DataNYU import *

import h5py
    
class NYU14(data.Dataset):
    def __init__(self, root='../data/nyu14/', task='train', split=1.0,
                nUseJoint=31):
        """
        INPUT
        - split: the ratio of splitting the dataset
                 1, no split
        """
        
        self.root = root
        self.split = split
        self.nUseJoint = nUseJoint
        self.task = task
        
        if os.path.exists(os.path.join(root, 'h5data')) == False:
            makeH5(root)
        
        if task == 'train':
            h5 = h5py.File(root+'/h5data/train_0.h5')
        elif task == 'test1':
            h5 = h5py.File(root+'/h5data/test_1_0.h5')
        elif task == 'test2':
            h5 = h5py.File(root+'/h5data/test_2_0.h5')
            
        print('data loading...')
        # it is critical to add [()], See
        # http://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
        self.depths = h5['depth'][()]
        self.joints = h5['joint'][()]
        self.coms = h5['com'][()]
        
        # if task == 'train':
        #     for i in range(1,3,1):
        #         print('data loading...')
        #         h5 = h5py.File(root+'/h5data/train_'+str(i)+'.h5')
        #         self.depths += list(h5['depth'])
        #         self.joints += list(h5['joint'])
        #         self.coms += list(h5['com'])
        #         self.inds += list(h5['inds'])
        
        print('data loaded, size: ', len(self.depths))
        
        self.length = (int)(len(self.depths) * split)
        
        # if self.limit_joints:
        #     self.eval_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        # else:
        #     self.eval_idxs = np.arange(36)
        # self.num_joints = len(self.eval_idxs)
        
    # def getNumJoints(self):
    #     return self.nJoints
        
    def __getitem__(self, idx):
        depth = self.depths[idx]
        joint = self.joints[idx][0:3*self.nUseJoint]
        com   = self.coms[idx]
        
        return torch.from_numpy(depth), torch.from_numpy(joint), torch.from_numpy(com)
    
    def __len__(self):
        return self.length
    
def convertNormTo3D(X):
    int((X[0] + 1) / 2 * 128), int( (-X[0]+1)/2 * 128)
    
# class ListDataset(data.Dataset):
#     def __init__(self, root, path_list, transform=None, target_transform=None,
#                  co_transform=None, loader=default_loader):

#         self.root = root
#         self.path_list = path_list
#         self.transform = transform
#         self.target_transform = target_transform
#         self.co_transform = co_transform
#         self.loader = loader

#     def __getitem__(self, index):
#         inputs, target = self.path_list[index]

#         inputs, target = self.loader(self.root, inputs, target)
#         if self.co_transform is not None:
#             inputs, target = self.co_transform(inputs, target)
#         if self.transform is not None:
#             inputs[0] = self.transform(inputs[0])
#             inputs[1] = self.transform(inputs[1])
#         if self.target_transform is not None :
#             target = self.target_transform(target)
#         return inputs, target

#     def __len__(self):
#         return len(self.path_list)