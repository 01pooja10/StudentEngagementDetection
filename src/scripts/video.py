import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
import shutil
import warnings
warnings.filterwarnings('ignore')

count = 0
skip = 30
f = 0

video_folder = 'E:/Datasets/Profali/DAISEE'
ids = os.listdir(video_folder)
new = 'E:/Datasets/Profali/Train'
#print(ids[1:3])

for i in ids[1:3]:
	#clips = os.path.join(video_folder, i)
	clips = video_folder+'/'+i
	
	for vids in tqdm(os.listdir(clips)):
	
		vids_path = clips + '/' + vids
		mp4 = vids + '.avi'
		video = vids_path + '/' + mp4
		#print(video)
		h_id = video.split('.')[0].split('/')[-1]
		to_save = new + '/' + h_id
		print(to_save)
		'''
  		os.mkdir(to_save)
  
	
		cap = cv2.VideoCapture(video)
		f = 0
		while True:
			count += 1
			ret = cap.grab()
			if count%skip == 0:
				ret, frame = cap.retrieve()
				if ret:
					frame = cv2.resize(frame, (128,128))
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					cv2.imwrite(to_save + '/frame' + str(f) + '.jpg', frame)
					f += 1
				else:
					break
				
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
   
cap.release()
cv2.destroyAllWindows()
				
print('done')
'''