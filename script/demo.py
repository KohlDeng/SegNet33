#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 23:49:55 2022

@author: dian
"""


import os
import argparse

print(os.getcwd())

pr = argparse.ArgumentParser(description='Pytorch SegNet example')
pr.add_argument('--lr', type=float,default=0.01,help='learning rate (default: 0.01)')

parser = argparse.ArgumentParser(description='PyTorch SegNet example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='training batch-size (default: 12)')
parser.add_argument('--epochs', type=int, default=300, metavar='E', help='no. of epochs to run (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='MOM', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--in-chn', type=int, default=3, metavar='IN', help='input image channels (default: 3 (RGB Image))')
parser.add_argument('--out-chn', type=int, default=32, metavar='OUT', help='output channels/semantic classes (default: 32)')

hyperparam = parser.parse_args()

hp = pr.parse_args()
print('learning rate is ',hp.lr)

print(pr.description)