
import os

import torch
import matplotlib.pyplot as plt
import skimage.io as io
import torchvision.transforms as transforms
import numpy as np

from segnet import SegNet
from camvid import CamVid

raw_dir = '/home/dian/Documents/dataset/CamVid/raw'
lbl_dir = '/home/dian/Documents/dataset/CamVid/label'
classes = np.load('classes.npy')
path = '/home/dian/Documents/vscode/python/SegNet'

model = SegNet(in_chn=3,out_chn=32)
weight_file = os.path.join(path,'checkpoint/segnet_weight_epoch_5.pth.tar')
# print(weight_file)
if os.path.isfile(weight_file): 
    print("Loading checkpoint '{}'".format(weight_file))
    checkpoint = torch.load(weight_file)
    epoch = checkpoint['epoch']
    print("epoch of weight file is:",epoch)
    
    # model.load_state_dict(checkpoint['state_dict'])
    # print(checkpoint[''])
else:
    print("no weights file")


transform = transforms.Compose([transforms.ToTensor()])
dataset = CamVid(classes,raw_dir,lbl_dir,transform=transform)


# tensor = torch.tensor(image_raw,dtype=float).permute(2,0,1)
data = dataset.__getitem__(256)
image_raw,label = data
(C,H,W)=image_raw.shape
tensor = torch.zeros(1,C,H,W)
tensor[0] = image_raw
output = model.forward(tensor)
print(output.size())
# plt.show()

