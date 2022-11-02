#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 23:49:55 2022

@author: dian
"""

import torch
from torchsummary import summary
from segnet import SegNet
from torchvision import models
# model = torch.load("/home/dian/Documents/spytorch/SegNet/checkpoint/my_model.pth")
model =SegNet(3, 32,momentum=0.6)
device = 'cuda:0'
input = torch.rand(1,3,224,224)
# model = models.resnet18()

print(torch.__version__)

summary(model.to(device),input_size=(3,224,224))

# torch.onnx.export(model.to(device), input.to(device),"./checkpoint/my_model.onnx",export_params=True)

# torch.save(model,"./checkpoint/my_model.onnx")
























































# print(output.shape)