import torch.nn as nn
import torch.nn.functional as F

# See Figure 1 in oberweger2015hands

class View(nn.Module):
    def __init__(self, o):
        super(View, self).__init__()
        self.o = o
        
    def forward(self, x):
        # print(x.size())
        return x.view(-1, self.o)

class DeepPriorSimple(nn.Module):
    def __init__(self, nUseJoint=31):
        super(DeepPriorSimple, self).__init__()
        
        self.m = nn.Sequential(
            nn.Conv2d(1, 8, padding=2, kernel_size=5),
            nn.MaxPool2d(4),
            nn.ReLU(True),
            # nn.Dropout(0.25),13
            # nn.Conv2d(8, 8, padding=2, kernel_size=5),
            View(8192),
            nn.Linear(8192, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 3*nUseJoint),
        )

    def forward(self, x):
        return self.m(x)

class zhou2016(nn.Module):
    def __init__(self, nUseJoint=31):
        super(zhou2016, self).__init__()
        
        self.m = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.MaxPool2d(4),
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=5),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 8, kernel_size=3),
            View(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 3*nUseJoint),
        )
        
    def forward(self, x):
        return self.m(x)
    
class DeepPriorDeep(nn.Module):
    def __init__(self, nUseJoint=31):
        super(DeepPriorDeep, self).__init__()
        
        self.m = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.MaxPool2d(3),
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=5),
            nn.MaxPool2d(3),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 8, padding=2, kernel_size=5),
            View(1568),
            nn.Linear(1568, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 3*nUseJoint),
        )

    def forward(self, x):
        return self.m(x)
    
class DeepPriorBottle(nn.Module):
    def __init__(self, nUseJoint=31, dimEmbedding=8):
        super(DeepPriorBottle, self).__init__()
        
        self.m = nn.Sequential(
            nn.Conv2d(1, 8, padding=2, kernel_size=5),
            nn.MaxPool2d(3),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            # nn.Dropout(0.3),
            nn.Conv2d(8, 8, padding=2, kernel_size=5),
            nn.MaxPool2d(3),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            # nn.Dropout(0.3),
            nn.Conv2d(8, 8, padding=2, kernel_size=5),
            View(1568),
            nn.Linear(1568, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # nn.Dropout(0.3),
            nn.Linear(1024, dimEmbedding),
            # nn.BatchNorm1d(dimEmbedding),
            nn.ReLU(True),
            nn.Linear(dimEmbedding, 3*nUseJoint),
        )

    def forward(self, x):
        return self.m(x)
    
# class DeepNetScale(nn.Module):
#     def __init__(self, nUseJoint=31):
#         super(DeepNetDeep, self).__init__()
        
#         self.m = nn.Sequential(
#             nn.Conv2d(1, 8, padding=2, kernel_size=5),
#             nn.MaxPool2d(3),
#             nn.ReLU(True),
#             nn.Conv2d(8, 8, padding=2, kernel_size=5),
#             nn.MaxPool2d(3),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Conv2d(8, 8, padding=2, kernel_size=5),
#             View(1568),
#             nn.Linear(1568, 1024),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 3*nUseJoint),
#         )

#     def forward(self, x):
#         return self.m(x)