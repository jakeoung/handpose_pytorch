
# coding: utf-8

# In[39]:

# original code by github.com/devnag

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os

import torch.optim as optim
from datasets.nyu14 import *
from models.oberweger2015hands import *
from torch.autograd import Variable

import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'

if os.sys.argv[0].find('ipykernel') >= 0:
    useIpynb = 1
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
    get_ipython().magic(u'matplotlib inline')
    prog_eg = 'main.py'
    os.sys.argv = prog_eg.split(' ')
else:
    useIpynb = 0


# In[52]:

#####################################################################
## Parameters
#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--niter', default=100, type=int)
parser.add_argument('-b', '--batchSize', default=128, type=int)
parser.add_argument('--lr', default=0.5, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--log', default=50, type=int)
parser.add_argument('--nUseJoint', default=31, type=int)
parser.add_argument('--dimEmbedding', default=8, type=int)

args = parser.parse_args()
# print(args)

torch.manual_seed(1)

if args.gpu >= 0:
    args.cuda = True


# In[53]:

#####################################################################
## Data
#####################################################################
# kwargs = {'num_workers':2, 'pin_memory':True} if args.cuda else {}
kwargs = {'num_workers':2}

nyu_train = NYU14('../data/nyu14/', task='train', nUseJoint=args.nUseJoint, split=1.0)
trainLoader = torch.utils.data.DataLoader(nyu_train, batch_size=args.batchSize, shuffle=True, **kwargs)

nyu_test1 = NYU14('../data/nyu14/', task='test1', nUseJoint=args.nUseJoint)
testLoader = torch.utils.data.DataLoader(nyu_test1, batch_size=args.batchSize, shuffle=False, **kwargs)


# In[54]:

#####################################################################
## Model
#####################################################################
# net = DeepNetDeep()
# net = DeepNetDeep(args.nUseJoint)
# net = DeepNetSimple(nUseJoint=args.nUseJoint)
net = DeepNetBottle(args.nUseJoint, dimEmbedding=args.dimEmbedding)
# net = zhou2016(args.nUseJoint)
if args.cuda:
    net.cuda()


# In[55]:

#####################################################################
## Optimize
#####################################################################
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)


# In[56]:

#####################################################################
## Task
#####################################################################
def train(epoch):
    net.train()
    epoch_loss = 0.
    for batch_idx, (data, target, _) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        
        pred = net(data)
        
        loss = F.smooth_l1_loss(pred, target)
#         loss = 0.5*torch.mean( (pred - target)**2 )
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.data[0]
        if batch_idx % args.log == 0:
            print('Train Epoch: {} [{:6d}/{:6d} ({:.6f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss.data[0]))

    epoch_loss /= len(trainLoader)
    return epoch_loss

def test(epoch):
    net.eval()
    test_loss = 0.

    for data, target, _ in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data, volatile=True), Variable(target)
        pred = net(data)
        
        loss = F.smooth_l1_loss(pred, target).cpu()
#         loss = torch.mean( (pred - target)**2 )
#         print(loss)
        test_loss += loss.data[0]

#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(testLoader)
    return test_loss


# In[57]:

trainLoss, testLoss = [], []

for epoch in range(args.niter):
    trainLoss_ = train(epoch)
    testLoss_ = test(epoch)
    print('epoch loss: {}'.format(trainLoss_))
    print('test loss: ', testLoss_)
    
    trainLoss.append( trainLoss_ )
    testLoss.append( testLoss_ )
    
    if epoch % 20 == 0:
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * 0.05


# In[46]:

#####################################################################
## Save result
#####################################################################
import time

fResult = time.strftime('%m%d_')
dic = vars(args)
key_params = dic.keys()
# key_params.sort()
for key in key_params:
    fResult += '--' + str(key) + '_' + str(dic[key]) + '_'
    
np.save('../result/'+fResult, [trainLoss, testLoss])


# In[47]:

#####################################################################
## Visualization
#####################################################################
plt.plot(trainLoss, label='train')
plt.plot(testLoss, label='test')
plt.legend()
plt.yscale('log')
plt.show()


# In[48]:

fx = 588.03
fy = 587.07
fx = 588.036865;
fy = 587.075073;

def showJoints2D(pred, com):
    x,y = joint2img(pred, com)
    plt.scatter(x,y)
#     for i in range(len(pred_img_coord)):
#         plt.scatter(pred_img_coord[i][0], pred_img_coord[i][1])

def jointToXYZ(joint):
    nPoints = (int)(len(joint)/3)
    x = np.zeros(int(nPoints), dtype=np.float32)
    y = np.zeros(int(nPoints), dtype=np.float32)
    z = np.zeros(int(nPoints), dtype=np.float32)
    cnt=0
    
    for i in range(0,len(joint),3):
        x[cnt] = joint[i]*150. + com[0]
        y[cnt] = joint[i+1]*150. + com[1]
        z[cnt] = joint[i+2]*150. + com[2]
        cnt += 1

    return x,y,z

def joint2img(joint, com):
    nPoints = (int)(len(joint)/3)
    x = np.zeros(int(nPoints), dtype=np.float32)
    y = np.zeros(int(nPoints), dtype=np.float32)
    u = np.zeros(int(nPoints))
    v = np.zeros(int(nPoints))

    z = np.zeros(int(nPoints))
    cnt=0
    
#     com = np.zeros(3)
    #step1: unnormalized joint images
    for i in range(0,len(joint),3):
        x[cnt] = joint[i]*150. + com[0]
        y[cnt] = joint[i+1]*150. + com[1]
        z[cnt] = joint[i+2]*150. + com[2]
        cnt += 1

    #step2: projection
    for i in range(nPoints):
        u[i] = (int)(x[i]*fx / z[i] + 320.)
        v[i] = (int)(-y[i]*fy / z[i] + 240.)
#         v[i] = (int)(y[i]*fy / z[i] + 240.)

#     for i in range(nPoints):
#         u[i] = x[i]*fx / z[i] + 128
#         v[i] = -y[i]*fy / z[i] + 64

        
#     for i in range(0,nPoints):
#         x[i] = joint[i*3]*150. + com[0]
#         y[i] = joint[i*3+1]*150. + com[1]
#         u[i], v[i] = convertTo2D(x[i], y[i])

#          u[i], v[i] = convertTo2D(joint[i*3], joint[i*3+1])

        
    return u,v

def convertTo2D(x, y):
    return int((x+1)/2. * 128), int( (-y+1)/2. * 128)

def showJoints3D(pred):
#     pred_img_coord = joint2img(pred)
    x,y,z = joint2imgOri(pred)
    plt.scatter(z,-x,y)
        
def joint2imgOri(joint):
    x = np.zeros(int(len(joint)/3))
    y = np.zeros(int(len(joint)/3))
    z = np.zeros(int(len(joint)/3))
    cnt=0
    for i in range(0,len(joint),3):
        x[cnt]=joint[i]
        y[cnt]=joint[i+1]
        z[cnt]=joint[i+2]
        cnt+=1
        
    return x,y,z


# In[51]:

for depth, joint, com in testLoader:
    if args.cuda:
        depth_cuda = depth.cuda()

    pred = net(Variable(depth_cuda))
    pred = pred.cpu().data.numpy()
    print(pred.shape)
    
    idx = 88
    fName = '../data/nyu14/test/depth_1_{:07d}.png'.format(idx+1)
    img_ori = plt.imread(fName)
    plt.imshow(img_ori[:,:,1], cmap=plt.cm.gray)

    showJoints2D(joint[idx,:], com[idx,:].numpy())
    plt.show()
    
    plt.imshow(img_ori[:,:,1], cmap=plt.cm.gray)
    showJoints2D(pred[idx,:], com[idx,:].numpy())
    plt.show()
    break


# plt.imshow(depth)


# In[36]:

for depth, joint, com in trainLoader:
    if args.cuda:
        depth_cuda = depth.cuda()

    pred = net(Variable(depth_cuda))
    pred = pred.cpu().data.numpy()
    
    idx = 111
    fName = '../data/nyu14/train/depth_1_{:07d}.png'.format(idx+1)
    img_ori = plt.imread(fName)

    plt.imshow(img_ori[:,:,1], cmap=plt.cm.gray)

    showJoints2D(joint[idx,:], com[idx,:].numpy())
    plt.show()
    
    plt.imshow(img_ori[:,:,1], cmap=plt.cm.gray)
    showJoints2D(pred[idx,:], com[idx,:].numpy())
    plt.show()
    break


# In[ ]:

####### 3D plot
from mpl_toolkits.mplot3d import Axes3D

Edges = [[0, 1], [1, 2], [2, 3], [3, 4], 
         [5, 6], [6, 7], [7, 8], [8, 9],
         [10, 11], [11, 12], [12, 13], [13, 14], 
         [15, 16], [16, 17], [17, 18], [18, 19],
         [4, 20], [9, 21], [14, 22], [19, 23], 
         [20, 24], [21, 24], [22, 24], [23, 24],
         [24, 25], [24, 26], [24, 27],
         [27, 28], [28, 29], [29, 30]]

x,y,z = jointToXYZ(joint)

fig=plt.figure()
ax=fig.add_subplot((111), projection='3d')
ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')
ax.scatter(z, -x, y)
for e in Edges:
    ax.plot(z[e], -x[e], y[e], c = 'b')
    
#For axes equal
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([zb], [xb], [yb], 'w')
    
plt.show()

