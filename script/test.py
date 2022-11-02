from camvid import CamVid
import os
import torch
import numpy as np

print("All required modules imported, loading data...")


raw_dir = '/home/dian/Documents/spytorch/SegNet/CamVid/raw'
lbl_dir = '/home/dian/Documents/spytorch/SegNet/CamVid/label'
classes = np.load('classes.npy')

cam_vid = CamVid(classes,raw_dir,lbl_dir)

cam_vid_loader = torch.utils.data.DataLoader(cam_vid, batch_size=5, shuffle=True, num_workers=4)

print("Data successfully loaded, enumerating...")

for i, data in enumerate(cam_vid_loader, 1):

	image, label = data
	print('Data no.', i, '| Raw image dimensions:', image.shape, '| Label dimensions:', label.shape)