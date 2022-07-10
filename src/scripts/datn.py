import numpy as np
import cv2
from PIL import Image
import os
import time
import logging
import pickle
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import shutil
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#initializing paths
subject = 'E:/Datasets/Profali/DAISEE/'
s1 = '500044/'

#initializing variables
store = []
count = 0
skip = 30
cb = 0
cc = 0
ce = 0
cf = 0

print('Initiliazing frame array')
frames = np.zeros((119, 10, 128, 128, 3))
d0 = 0
d1 = 0

store = os.listdir(subject+s1)
path = []
for i in range(len(store)):
    path.append(subject+s1+str(store[i]))
print('Size of list:', len(path))

#checking for path consistency
#print(path[0])
#print(str(path[0].split('/')[-1]) + '.avi')

for j in tqdm(path):
	spath = j + '/'
	modpath = str(j.split('/')[-1]) + '.avi'
    #print(spath + modpath)
	cap = cv2.VideoCapture(spath + modpath)
	
	d1 = 0
	while True:
		count += 1
        
		ret = cap.grab()
        #print(ret)
		if count % skip == 0:
			ret, frame = cap.retrieve()
            
			if ret:
                #cv2.imshow('vid', frame)
				frame = cv2.resize(frame, (128,128))
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                trans = transforms.Compose([transforms.ColorJitter(0.3, 0.2),
                                            transforms.ToTensor(),
                                           transforms.Normalize((0.5,),(0.5,))
                                           ])
                frame = trans(frame)
                frame = frame.detach().numpy()
				frames[d0,d1,:,:,:] = frame
				d1 += 1
                
			else:
				break
            
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
    
	d0 += 1
                

cap.release()
cv2.destroyAllWindows()

#torch.Size([1190, 3, 128, 128])
#print(frames[49])

framet = torch.FloatTensor(frames)
print(framet.size())

with open('fram500044.pkl', 'wb') as fl:
    pickle.dump(framet, fl)
    
