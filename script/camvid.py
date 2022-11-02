#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 09:09:25 2022

@author: dian
waiting understanding
from torch.utils.data import Dataset
label = np.nanmin(self.classes[c]==image,axis=2)
classes 32 label
"""

import torch
from torch.utils.data import Dataset
import os
from skimage import io
import time
import numpy as np

class CamVid(Dataset):
    
    def getLabeled(self,img_name,lbl_dir):
        index=img_name.find('.png')
        img_lbl_dir = os.path.join(lbl_dir,(img_name[:index]+'_L'+img_name[index:]))
        
        return img_lbl_dir
    
    def __init__(self,classes,raw_dir,lbl_dir,transform=None):
        
        self.classes = classes
        self.raw_dir = raw_dir
        self.lbl_dir = lbl_dir
        self.transform = transform
        self.list_img = os.listdir(self.raw_dir)
        
    def one_hot(self,image):
        output_shape = (image.shape[0],image.shape[1],self.classes.shape[0])
        output = np.zeros(output_shape)
        
        for c in range(self.classes.shape[0]):
            #像素点挨个比较
            label = np.nanmin(self.classes[c]==image,axis=2)
            output[:,:,c] = label
        
        return output
    
    def __len__(self):
        return len(self.list_img)
    
    def __getitem__(self,idx):
        img_raw_name = self.list_img[idx]
        img_raw_dir = os.path.join(self.raw_dir,img_raw_name)
        image_raw = io.imread(img_raw_dir)
        img_lbl_dir = self.getLabeled(img_raw_name,self.lbl_dir)
        image_label = io.imread(img_lbl_dir)
        label = self.one_hot(image_label)
        
        if self.transform:
            image_raw = self.transform(image_raw)
            label = self.transform(label)
        data = (image_raw,label)
        return data   
        

              
   
        
# raw_dir = '/home/dian/Documents/dataset/CamVid/raw'
# lbl_dir = '/home/dian/Documents/dataset/CamVid/label'
# classes = np.load('classes.npy')

# dataset = CamVid(classes,raw_dir,lbl_dir)



##

# list_img = os.listdir(raw_dir)

# size = classes.shape

# print('classes is \n',classes)    
# print('size is ', classes.shape)

# img_raw_name = '0006R0_f01920.png'

    
# img_raw_dir = os.path.join(raw_dir,img_raw_name) 
# img_raw = io.imread(img_raw_dir)

# index = img_raw_name.find('.png')
# img_lbl_dir = os.path.join(lbl_dir,(img_raw_name[:index]+'L'+img_raw_name[index:]))


# print(img_raw.shape)    
    
    
    
    
    
    
    