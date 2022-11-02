#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May 20 22:21:24 2022

@author: dian
#waiting understanding
transform = transforms.Compose([transforms.ToTensor()])
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from segnet import SegNet
from camvid import CamVid

warnings.filterwarnings("ignore")

raw_dir = '/home/dian/Documents/dataset/CamVid/raw'
lbl_dir = '/home/dian/Documents/dataset/CamVid/label'
prj_dir = '/home/dian/Documents/vscode/python/SegNet/'

def main():
    classes = np.load('classes.npy')
    device = 'cpu'
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Training settings
    
    parser = argparse.ArgumentParser(description='PyTorch SegNet example')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='training batch-size (default: 12)')
    parser.add_argument('--epochs', type=int, default=9, metavar='E', help='no. of epochs to run (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='MOM', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--in-chn', type=int, default=3, metavar='IN', help='input image channels (default: 3 (RGB Image))')
    parser.add_argument('--out-chn', type=int, default=32, metavar='OUT', help='output channels/semantic classes (default: 32)')
    parser.add_argument('--weight',type=str,default="checkpoint/segnet_weight.pth.tar",help="weight file name(default:checkpoint/segnet_weight.pth.tar)")
        
    hyperparams = parser.parse_args()
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    trainset =CamVid(classes,raw_dir,lbl_dir,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hyperparams.batch_size, shuffle=True, num_workers=8)
    model = SegNet(3,32).to(device)
    path = os.path.join(prj_dir,hyperparams.weight)
    
    resume = input("Resume training? (Y/N): ")
   
    if resume == 'Y':
        if os.path.isfile(path):
            train(model,trainloader,hyperparams,device,path)
        else:
            print("weight file doesn't exit")
    elif resume == 'N':
        train(model,trainloader,hyperparams,device)
    else:
        print("Invalid input, exiting program.")
        return 0
                
    
def save_checkpoint(state,path):
    torch.save(state,path)
    print("check point saved at {}".format(path))


def train(model,trainloader,hyperparams,device,path=None):
    
    optimizer = optim.SGD(model.parameters(),lr=hyperparams.lr,momentum=hyperparams.momentum)
    loss_fn = nn.CrossEntropyLoss()
    epochs = hyperparams.epochs
    batch_size = hyperparams.batch_size        
    if path==None:
        run_epoch = 0   
        list_epoch=[]
        list_loss=[]
    else:  
        print("Loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        run_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
        list_epoch = checkpoint['list_epoch']
        list_loss = checkpoint['list_loss']

    
    print("start training...")     
    for epoch in range(1+run_epoch,epochs+1+run_epoch):  

        sum_loss = 0.0
        count = 0 #for demo test
        for j,data in enumerate(trainloader,1):
            images,labels = data
            
            optimizer.zero_grad()
            output = model(images.to(device))
            loss = loss_fn(output,labels.to(device))
            loss.backward()
            optimizer.step()
            
            sum_loss+=loss.item()
            count+=1
            if(count>=1):
                break
            # print('Loss at {} mini-batch: {}'.format(j, loss.item()/trainloader.batch_size))
        num_batch = count
        print("-----------------------------------------------------")
        list_epoch.append(epoch)
        list_loss.append(sum_loss/batch_size/num_batch)
        print("epoch {} is over,num_batch is {},loss is {}".format(epoch,num_batch,sum_loss/batch_size/num_batch))
        if(epoch%4==0):    
            print("Saving checkpoint...")
            weight_name = "checkpoint/segnet_weight_epoch_"+str(epoch)+".pth.tar"
            weight_path = os.path.join(prj_dir, weight_name)
            save_checkpoint({'epoch': epoch,'list_epoch':list_epoch,'list_loss':list_loss,'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, weight_path)
    
    print("-----------------------------------------------------")
    print("training is over")  
    weight_name = "checkpoint/segnet_weight_epoch_"+str(epoch)+".pth.tar"
    weight_path = os.path.join(prj_dir, weight_name)
    save_checkpoint({'epoch': epoch,'list_epoch':list_epoch,'list_loss':list_loss,'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, weight_path)
      
    array_epoch = np.array(list_epoch)
    array_loss = np.array(list_loss)
    plt.figure(1)
    plt.title("Loss During Training")
    plt.plot(array_epoch,array_loss,label="loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    figure_dir = os.path.join(prj_dir,"checkpoint/train_loss.jpg")
    plt.savefig(figure_dir, bbox_inches='tight')
    

        
if __name__ =='__main__':
    main()    
    