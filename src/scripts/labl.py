import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import os

c = 0

lab_path = 'E:/Datasets/Profali/DAISEE/1DATAINFO/TestLabels.csv'
#print(len(os.listdir('E:/Datasets/Profali/DAISEE/500044/')))
newp = 'E:/Datasets/Profali/D2/train'
df = pd.read_csv(lab_path)
print(df.head(122))
#print(df.iloc[:121])
reduce = df.iloc[:120]
l1 = reduce.values.tolist()
#print(l1)
#print(torch.FloatTensor(l1[0][1:]).size())
#print(l1[0:5])
labels = []
for i in l1:
    labels.append(i[1:])
'''
labels[labels==2] = 1
labels[labels==3] = 1
labels = F.one_hot(torch.from_numpy(labels).to(torch.int64),num_classes=4)


    for p in i[1:]:
        c += 1
        labels.append(p)
        if c%4 == 0:
            break
            '''
            
labf = torch.FloatTensor(labels)

print(labf.size())
'''
with open('ltensors.pkl', 'rb') as fl:
    pickle.load(labf,fl)
    '''

