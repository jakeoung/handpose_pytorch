# this file is modified from the original code:
# https://github.com/xingyizhou/DeepModel

import numpy as np
import h5py
import scipy.io as sio
import scipy.misc as misc
import sys
import os
import math

from skimage.transform import resize

from PIL import Image

import cv2

## This part of code is modified from [DeepPrior](https://cvarlab.icg.tugraz.at/projects/hand_detection/)
def CropImage(depth, com, cube_size):
    u, v, d = com
    zstart = d - cube_size / 2.
    zend = d + cube_size / 2.
    
    # pricinal points are omitted (due to simplicity?)
    xstart = int(math.floor((u * d / fx - cube_size / 2.) / d * fx))
    xend = int(math.floor((u * d / fx + cube_size / 2.) / d * fx))
    ystart = int(math.floor((v * d / fy - cube_size / 2.) / d * fy))
    yend = int(math.floor((v * d / fy + cube_size / 2.) / d * fy))
  
    cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
    cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, depth.shape[0])), 
                                (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, depth.shape[1]))), mode='constant', constant_values=0)
    msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
    msk2 = np.bitwise_and(cropped > zend, cropped != 0)
    cropped[msk1] = zstart
    cropped[msk2] = zend

    dsize = (img_size, img_size)
    wb = (xend - xstart)
    hb = (yend - ystart)
    if wb > hb:
        sz = (dsize[0], (int)(hb * dsize[0] / wb))
    else:
        sz = ((int)(wb * dsize[1] / hb), dsize[1])

    roi = cropped
    
    
    rz = cv2.resize(cropped, sz)
    # maxmin = cropped.max() - cropped.min()
    # cropped_norm = (cropped - cropped.min()) / maxmin
    # rz = maxmin * resize(cropped_norm, sz, mode='reflect', preserve_range=True) + cropped.min()
    # rz = rz.astype(np.float32)

    ret = np.ones(dsize, np.float32) * zend
    xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
    xend = int(xstart + rz.shape[1])
    ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
    yend = int(ystart + rz.shape[0])
    ret[ystart:yend, xstart:xend] = rz

    return ret

def readDepth(path):
    """
    Note: In each depth png file the top 8 bits of depth are
    packed into the green channel and the lower 8 bits into blue.
    See http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm#download
    Ref: [1]
    """
    rgb = Image.open(path)
    print(rgb)
    r, g, b = rgb.split()

    r = np.asarray(r, np.int32)
    g = np.asarray(g, np.int32)
    b = np.asarray(b, np.int32)

    # dpt = b + g*256

    dpt = np.bitwise_or(np.left_shift(g, 8), b)
    imgdata = np.asarray(dpt, np.float32)
    return imgdata

##
J = 31
# joint_id = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32, 1, 2, 4, 7, 8, 10, 13, 14, 16, 19, 20, 22, 5, 11, 17, 23, 28])
joint_id = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 5, 11, 17, 23, 32, 30, 31, 28, 27, 25, 24])
img_size = 128

fx = 588.03
fy = 587.07
fu = 320.
fv = 240.

data_names = ['train', 'test_1', 'test_2']
cube_sizes = [300, 300, 300]
id_starts = [0, 0, 2440]
id_ends = [72756, 2440, 8252]
#id_ends = [727, 2440, 8252]
# num_packages = [3, 1, 1]
num_packages = [1, 1, 1]

def makeH5(root='../data/nyu14/'):

    for D in range(0, len(data_names)):
        data_name = data_names[D]
        cube_size = cube_sizes[D]
        id_start = id_starts[D]
        id_end = id_ends[D]
        chunck_size = (int)((id_end - id_start) / num_packages[D])

        task = 'train' if data_name == 'train' else 'test'
        data_path = '{}/{}'.format(root, task)
        label_path = '{}/joint_data.mat'.format(data_path)

        labels = sio.loadmat(label_path)
        joint_uvd = labels['joint_uvd'][0]
        joint_xyz = labels['joint_xyz'][0]

        cnt = 0
        chunck = 0
        depth_h5, joint_h5, com_h5, = [], [], []
        for idx in range(id_start, id_end):
            img_path = '{}/depth_1_{:07d}.png'.format(data_path, idx + 1)

            if not os.path.exists(img_path):
                print('{} Not Exists!'.format(img_path))
                continue

            print(img_path)
            depth = readDepth(img_path)

            # is joint_uvc[id, 34] center of mass???
            
            depth = CropImage(depth, joint_uvd[idx, 34], cube_size)

            com3D = joint_xyz[idx, 34]
            joint = joint_xyz[idx][joint_id] - com3D
            
            # normalize depth to [-1,1] and resize to one of the shape [128,128]
            depth = ((depth - com3D[2]) / (cube_size / 2)).reshape(1, img_size, img_size)

            # normalized ground truth joint 3d coordinates to [-1,1]
            joint = np.clip(joint / (cube_size / 2), -1, 1)
            depth_h5.append(depth.astype(np.float32))
            joint_h5.append(joint.astype(np.float32).reshape(3 * J))
            com_h5.append(com3D.copy())
            cnt += 1
            if cnt % chunck_size == 0 or idx == id_end - 1:
                dH5 = os.path.join(root, 'h5data/')
                try:
                    os.makedirs(dH5)
                except OSError:
                    pass
                
                # rng = np.arange(cnt) if task == 'test' else np.random.choice(np.arange(cnt), cnt, replace = False)
                dset = h5py.File((dH5+'/{}_{}.h5').format(data_name, chunck), 'w')
                dset['depth'] = np.asarray(depth_h5)
                dset['joint'] = np.asarray(joint_h5)
                dset['com'] = np.asarray(com_h5)
                dset.close()
                chunck += 1
                cnt = 0
            
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Specify the root directory of NYU14 directory as argument')
    else:
        makeH5(sys.argv[1])
