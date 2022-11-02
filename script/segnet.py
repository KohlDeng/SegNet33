#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May 20 22:21:24 2022

@author: dian
#waiting understanding
momentum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


# input 3x416x416
class SegNet(nn.Module):
    def __init__(self, in_chn, out_chn, momentum=0.5):
        super().__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn

        # Encoding
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True,ceil_mode=False)

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=momentum)

        # Decoding
        self.maxup = nn.MaxUnpool2d(2, stride=2)

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=momentum)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x)),inplace=True) 
        x = F.relu(self.BNEn12(self.ConvEn12(x)),inplace=True) 
        x, ind1 = self.maxpool(x)
        size1 = x.size()
        # print("ind1 size is:",ind1.size())
        # print("size1 is ",size1)
        # Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x)),inplace=True) 
        x = F.relu(self.BNEn22(self.ConvEn22(x)),inplace=True) 
        x, ind2 = self.maxpool(x)
        size2 = x.size()
        # print("size2 is ",size2)
        
        # Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x)),inplace=True) 
        x = F.relu(self.BNEn32(self.ConvEn32(x)),inplace=True) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)),inplace=True) 	
        x, ind3 = self.maxpool(x)
        size3 = x.size()
        # print("size3 is ",size3)
        # Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x)),inplace=True) 
        x = F.relu(self.BNEn42(self.ConvEn42(x)),inplace=True) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)),inplace=True) 
        x, ind4 = self.maxpool(x)
        size4 = x.size()
        # print("size4 is ",size4)
        
        # Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x)),inplace=True) 
        x = F.relu(self.BNEn52(self.ConvEn52(x)),inplace=True) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)),inplace=True) 
        x, ind5 = self.maxpool(x)
        size5 = x.size()
        # print("size5 is ",size5)

        # DeCoding
        
        x = self.maxup(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)),inplace=True)
        x = F.relu(self.BNDe52(self.ConvDe52(x)),inplace=True)
        x = F.relu(self.BNDe51(self.ConvDe51(x)),inplace=True)
        size6 = x.size()
        # print("size6 is ",size6)
        
        x = self.maxup(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)),inplace=True)
        x = F.relu(self.BNDe42(self.ConvDe42(x)),inplace=True)
        x = F.relu(self.BNDe41(self.ConvDe41(x)),inplace=True)
        size7 = x.size()
        # print("size7 is ",size7)

        x = self.maxup(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)),inplace=True)
        x = F.relu(self.BNDe32(self.ConvDe32(x)),inplace=True)
        x = F.relu(self.BNDe31(self.ConvDe31(x)),inplace=True)
        size8 = x.size()
        # print("size8 is ",size8)

        x = self.maxup(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)),inplace=True)
        x = F.relu(self.BNDe21(self.ConvDe21(x)),inplace=True)
        size9 = x.size()
        # print("size9 is ",size9)

        x = self.maxup(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)),inplace=True)
        # x = F.relu(self.BNDe11(self.ConvDe11(x)))
        x = self.ConvDe11(x)
        size10 = x.size()
        # print("size10 is ",size10)
        return x

# device = "cpu"

# input = torch.rand(5,3,224,224)
# model =SegNet(3,30,momentum=0.6).to(device)

# output = model.forward(input.to(device))
# print("output size is ", output.size())

# # print('Epoch {}'.format(2))

# torch.save(model,"checkpoint/my_model.pth")


        
    
    
    
    
    
    
    
    