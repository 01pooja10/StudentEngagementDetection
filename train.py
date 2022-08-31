import os
import cv2
import copy
import torch
import pickle
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.utils.prune as prune
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34, vgg16_bn

#check cuda availability
torch.cuda.get_device_name(0)

#User inputs
__modelnames__ = ['resnet503d', 'resnet342d', 'resnet182d', 'resnet183d', 'vgg162d']

@staticmethod
def ValidModels():
	print(__modelnames__)
 

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--lr", default=0.0001, type=int, required=True, help="learning rate value")
parser.add_argument("--epochs", default=150, type=int, required=True, help="number of iterations")
parser.add_argument("--frames", default=1, type=int, required=True, help="number of frames (2D/3D")
parser.add_argument("--model_name", default="resnet183d", type=str, required=True, help="name of the model")

args.parser.parse_args()

print(args.model_name)
if args.model_name not in __modelnames__:
    raise ValueError("Model not found. {} are the supported models".format(__modelnames__))


#Prepare dataloaders
img_path = input('Enter path to images stores as tensors.')
label_path = input('Enter path to labels stored as tensors.')

assert os.path.exists(img_path), 'File not found at'+ str(img_path)
assert os.path.exists(label_path), 'File not found at'+ str(img_path)

with open(img_path,'rb') as fl1:
    imgs = pickle.load(fl1)

with open(label_path,'rb') as fl2:
    labs = pickle.load(fl2)
	
	
lab = F.one_hot(labs.to(torch.int64), num_classes=2)
#print(labs.size())
#print(imgs.size())

def AllocateFrames(imgs):
	
	if args.frames == "1 frame":
		fs = np.zeros((1531,128,128,3))
		c = 0
		for i in range(len(imgs)):  
			fs[c,:,:,:] = imgs[i][5]
			c += 1
		fs = torch.FloatTensor(fs)
		fs1 = fs[:1384]
		return fs1

	elif args.frames == "3 frames":
		fs = np.zeros((1531,3,128,128,3))
		c = 0
		d = 0
		for i in range(len(imgs)):
			d = 0
			for j in range(4,7):
				fs[c,d,:,:,:] = imgs[i][j]
				d += 1
			c += 1
		fs = torch.FloatTensor(fs)
		fs3 = fs[:1384]
		return fs3

	elif args.frames == "5 frames":	
		fs = np.zeros((1531,5,128,128,3))
		c = 0
		d = 0
		for i in range(len(imgs)):
			d = 0
			for j in range(3,8):
				#print('init', i,j)
				#print(c,d)
				fs[c,d,:,:,:] = imgs[i][j]
				d += 1
			c += 1
		fs = torch.FloatTensor(fs)
		fs5 = fs[:1384]
		return fs5

	elif args.frames == "7 frames":
		fs = np.zeros((1531,7,128,128,3))
		c = 0
		d = 0
		for i in range(len(imgs)):
			d = 0
			for j in range(3,10):
				fs[c,d,:,:,:] = imgs[i][j]
				d += 1
			c += 1
		fs = torch.FloatTensor(fs)
		fs7 = fs[:1384]
		return fs7

mod = args.model_name

if mod ==  'resnet503d':
	model = ResNet503D(Bottleneck, [3, 4, 6, 3], 48, 32, 'B', 2, True).cuda()
	
elif mod ==  'resnet183d':
	model = resnet18(pretrained=False)
	res2d.fc = nn.Linear(512, 2)
	
elif mod ==  'resnet183d':
	model = resnet18(pretrained=False)
	model.fc = nn.Linear(512, 2)
	
elif mod ==  'resnet342d':	
	model = resnet34(pretrained=False)
	model.fc = nn.Linear(512, 2)
	
elif mod ==  'vgg162d':
	vgg2d = vgg16_bn()
	nf = vgg2d.classifier[6].in_features
	feats = list(vgg2d.classifier.children())[:-1]
	feats.extend([nn.Linear(nf, 2)])
	vgg2d.classifier = nn.Sequential(*feats)
	
	
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
