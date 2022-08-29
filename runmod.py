import os
import cv2
import time
import dlib
import torch
import pickle
import random
import logging
import asyncio
import numpy as np
import torch.nn as nn
import mediapipe as mp
from functools import partial
from imutils import face_utils
from torchvision import transforms
from multiprocessing import Pool
from torchvision.models.video import r3d_18
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as npc

import warnings
warnings.filterwarnings('ignore')

blinkcount = 0
totalblinks = 0
framecount = 0
eyecount = 0
c = 0
fps = 30
skip = 10
yawncount = 0
yawn_threshold = 23
eye_frames = 25



async def BehaviourAnalysis() -> str:
	
	path = 'E:/Conference_stuff/student/drive/net3d18_3f.pth'
	cascp = 'E:/Conference_stuff/student/drive/cascade.xml'
	facecasc = cv2.CascadeClassifier(cascp)
	
	resnet18_3f = torch.load(path)
	model = r3d_18(pretrained=False)
	model.fc = nn.Linear(512,2)
	model.load_state_dict(resnet18_3f['model'])
	model.eval()

	frames = np.zeros((3,128,128,3))
	flist = []
	c1 = 0
	status = ''
	cam = cv2.VideoCapture(0)
	while True:
		
		ret, fm = cam.read()
		
		if ret:
			
			frame = cv2.cvtColor(fm, cv2.COLOR_BGR2RGB)
			frame = cv2.resize(frame, (128,128))
			gr = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY)
			faces = facecasc.detectMultiScale(gr, 1.1, 5)
			for (x,y,w,h) in faces:
				cv2.rectangle(fm, (x,y), (x+w, y+h), (255,255,255),2)
			
			transf = transforms.ToTensor()
			framet = transf(frame)
			framet = torch.permute(framet, (1,2,0))
			
			if c1 < 3:
				#print(c)
				frames[c1,:] = framet
				c1 += 1
			else:
				c1 = 0
				frames = np.zeros((3,128,128,3))
				continue
				
				
		flist.append(frames)
		#print(torch.as_tensor(frames).size())
		
		#print(len(flist))
		if len(flist)%30 == 0:
			flt = torch.as_tensor(flist)
			#print(flt.size())
			with torch.no_grad():
				out = model(flt.float())
			#print(out)
			out = torch.sigmoid(out)
			#print(out[29])
			ch = out[29]
			print(ch)
			#print(ch)
			a = torch.argmax(ch)
			print(a)
			ch[ch == a] = 1
			ch[ch < a] = 0
			print(ch)
			if a == 0:
				print('Disengaged')
				status = 'Disengaged'
				
			else:
				print('Engaged')
				status = 'Engaged'
			flist = []	
		cv2.putText(fm, status, (30,110), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), 1)
		cv2.imshow('webcam', fm)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break
		
	cam.release()
	cv2.destroyAllWindows()
	return status

	
#print(len(flist[0]))


def Landmarks(img,results,draw=False):
	h, w = img.shape[:2]
	coords = np.array([(int(f.x * w),int(f.y * h)) for f in results.multi_face_landmarks[0].landmark])
	if draw:
		[cv2.circle(img,p,2,(0,255,0),-1) for p in coords]
	return coords


def Euclidean(p1,p2):
	x1,y1 = p1.ravel()
	x2,y2 = p2.ravel()
	sq1 = (x2-x1)**2
	sq2 = (y2-y1)**2
	euc = np.sqrt(sq1+sq2)
	return euc

def Position(centre, right, left):
	ctor = Euclidean(centre, right)
	total = Euclidean(right, left)
	ratio = (ctor/total)
	iris = ''
	if ratio <= 2.5:
		iris = 'Left' #right
	elif ratio>2.5 and ratio <= 3.25:
		iris = 'Center'
	else:
		iris = 'Right' #left

	return iris, ratio


def Blink(img, landmarks, ridx, lidx):
	#right eye horizontal
	rr = landmarks[ridx[0]]
	rl = landmarks[ridx[8]]

	#right eye vertical
	rt = landmarks[ridx[12]]
	rb = landmarks[ridx[4]]

	#cv2.line(img,rr,rl,(0,255,0),2)
	#cv2.line(img,rt,rb,(0,255,0),2)

	#left eye horizontal
	lr = landmarks[lidx[0]]
	ll = landmarks[lidx[8]]

	#left eye vertical
	lt = landmarks[lidx[12]]
	lb = landmarks[lidx[4]]

	#cv2.line(img,lr,ll,(0,255,0),2)
	#cv2.line(img,lt,lb,(0,255,0),2)

	#calculating euclidean dist b/w right & left, top & bottom - right eye
	rhdist = Euclidean(rr,rl)
	#print(rhdist)
	rvdist = Euclidean(rt,rb)

	#calculating euclidean dist b/w right & left, top & bottom - left eye
	lhdist = Euclidean(lr,ll)
	lvdist = Euclidean(lt,lb)

	#finding ratio of horizontal/vertical
	rratio = rhdist/rvdist
	lratio = lhdist/lvdist

	ratio = (rratio + lratio)/2
	return ratio


#function to extract eyes
def Eyes(img,right,left):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#find img dimensions
	dims = gray.shape
	#create blank mask for dims
	mask = np.zeros(dims,dtype=np.uint8)

	#draw eye polygons
	cv2.fillPoly(mask,[np.array(right,dtype=np.int32)],255)
	cv2.fillPoly(mask,[np.array(left,dtype=np.int32)],255)
	cv2.imshow('mask', mask)

	eye = cv2.bitwise_and(gray,gray,mask=mask)
	eye[mask==0]=155
	cv2.imshow('eyes', eye)

	rxmax = (max(right, key = lambda item:item[0]))[0]
	#print(rxmax)
	rxmin = (min(right, key = lambda item:item[0]))[0]
	rymax = (max(right, key = lambda item:item[1]))[1]
	rymin = (min(right, key = lambda item:item[1]))[1]
	#print(rymin)

	lxmax = (max(left, key = lambda item:item[0]))[0]
	lxmin = (min(left, key = lambda item:item[0]))[0]
	lymax = (max(left, key = lambda item:item[1]))[1]
	lymin = (min(left, key = lambda item:item[1]))[1]

	cropr = eye[rymin:rymax, rxmin:rxmax]
	cropl = eye[lymin:lymax, lxmin:lxmax]
	
	#cv2.imshow('r', cropr)
	#cv2.imshow('l', cropl)
	return cropr,cropl

def EyePosition(cropeye):
	h, w = cropeye.shape
	blur = cv2.GaussianBlur(cropeye,(9,9),0)
	med = cv2.medianBlur(blur, 3)
	#cv2.imshow('blurred', med)

	r, thresh = cv2.threshold(med, 130, 255, cv2.THRESH_BINARY)
	#creating 3 parts - right, center and left - identify where white part is
	pc = int(w/3)

	rightp = thresh[0:h, 0:pc]
	centerp = thresh[0:h, pc:pc+pc]
	leftp = thresh[0:h, pc+pc:w]

	right = np.sum(rightp==0)
	mid = np.sum(centerp==0)
	left = np.sum(leftp==0)
	eye = [right, mid, left]

	eyepos, color = PixelCount(eye)
	return eyepos, color

def PixelCount(eye):
	maxidx = eye.index(max(eye))
	pos = ''
	color = ''

	if maxidx == 0:
		pos = 'Left'
		#right
		#color = [utils.BLACK, utils.GREEN]
	elif maxidx == 1:
		pos = 'Center'
		#color = [utils.BLUE, utils.WHITE]
	elif maxidx == 2:
		pos = 'Right'
		#left
		#color = [utils.GRAY, utils.YELLOW]
	else:
		pos = 'Closed'
		#color = [utils.RED, utils.ORANGE]
	return pos, color


def yawn(landmarks, frame):
	top, bottom = [], []

	for i in range(50, 53):
		x = landmarks.part(i).x

	for i in range(61, 64):
		y = landmarks.part(i).y

	top.append((x,y))

	for i in range(56, 59):
		x1 = landmarks.part(i).x

	for i in range(65, 68):
		y1 = landmarks.part(i).y

	bottom.append((x1,y1))

	top_mean = np.mean(top, axis=0)
	bottom_mean = np.mean(bottom, axis=0)

	dist = abs(top_mean[1] - bottom_mean[1])
	return dist



async def FacialTracker():
	
	FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

	# lips indices for Landmarks
	LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
	LOW = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
	UP = [185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

	#drowsiness
	# Left eyes indices
	L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
	L_BROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
	L_IRIS = [474, 475, 476, 477]

	# right eyes indices
	R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
	R_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
	R_IRIS = [469, 470, 471, 472]

	# extreme iris coordinates
	L_LEFT = [33]
	L_RIGHT = [133]
	R_LEFT = [362]
	R_RIGHT = [263]


	#FACIAL TRACKING
	mesh = mp.solutions.face_mesh
	draw = mp.solutions.drawing_utils
	spec = draw.DrawingSpec(thickness=1,circle_radius=1)

	
	cascp = 'E:/Conference_stuff/student/drive/cascade.xml'
	binfl = 'C:/Users/Pooja/code/mitacs/imgs/shape_predictor_68_face_landmarks.dat'
	facecasc = cv2.CascadeClassifier(cascp)

	blinkcount = 0
	totalblinks = 0
	framecount = 0
	eyecount = 0
	c = 0
	fps = 30
	skip = 10
	yawncount = 0
	yawn_threshold = 23
	eye_frames = 25

	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor(binfl)

	(mstart,mend) = face_utils.FACIAL_LANDMARKS_IDXS['inner_mouth']
	
	cam = cv2.VideoCapture(0)

	with mesh.FaceMesh(max_num_faces=1,
						refine_landmarks=True,
						min_detection_confidence=0.5,
						min_tracking_confidence=0.5) as fm:

		start = time.time()
		while True:
			framecount += 1
			ret, frame = cam.read()
			#frame = cv2.flip(frame,1)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
			frame = cv2.flip(frame,1)
			if not ret:
				break
			results = fm.process(frame)
			#frame.flags.writeable = True
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			if results.multi_face_landmarks:
				#change to false
				meshc = Landmarks(frame, results, False)
				ratio = Blink(frame, meshc, R_EYE, L_EYE)
				#to remove
	
				cv2.putText(frame,f'Ratio: {ratio: 0.2f}',(30,50),cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255),1)

				if ratio>5.5:
					blinkcount = blinkcount+1
					eyecount += 1
					#to remove - frame count = {bc}
					#cv2.putText(frame, 'Blink', (50,100), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), 1)
					if eyecount>=eye_frames:
						cv2.putText(frame, 'You seem drowsy', (30,110), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), 1)

				else:
					eyecount = 0
		
					faces = detect(frame,0)
					for face in faces:
						landmarks = predict(frame, face)
						landmarks = face_utils.shape_to_np(landmarks)
						mth = landmarks[mstart:mend]
						#mouth = cv2.convexHull(mth)
						#mouth = cv2.convexHull(meshc[LOW:UP])
						#toremove
						#cv2.drawContours(frame,[left_hull],-1,(0,255,0),1)
						#cv2.drawContours(frame,[right_hull],-1,(0,255,0),1)
						#cv2.drawContours(frame,[mouth],-1,(0,255,0),1)

						p = face.left()
						q = face.top()
						r = face.right()
						s = face.bottom()
						landmarks = predict(frame, face)

						y = yawn(landmarks,frame)
						if y > yawn_threshold:
							yawncount += 1
							cv2.putText(frame,f"YAWN ALERT!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
							yawncount = 0
						
					cv2.polylines(frame,[meshc[LOW]],True,(0,255,0),1,cv2.LINE_AA)
					cv2.polylines(frame,[meshc[UP]],True,(0,255,0),1,cv2.LINE_AA)
		
					(lx,ly), lrad = cv2.minEnclosingCircle(meshc[L_IRIS])
					(rx,ry), rrad = cv2.minEnclosingCircle(meshc[R_IRIS])
					#cv2.circle(frame, np.array([lx,ly], dtype = np.int32), int(lrad),(0,255,0),1)
					#cv2.circle(frame, np.array([rx,ry], dtype = np.int32), int(rrad),(0,255,0),1)
					#cv2.polylines(frame,[meshc[L_IRIS]],True,(0,255,0),1,cv2.LINE_AA)
					#cv2.polylines(frame,[meshc[R_IRIS]],True,(0,255,0),1,cv2.LINE_AA)

					lcentre = np.array([lx, ly], dtype=np.int32)
					rcentre = np.array([rx, ry], dtype=np.int32)

					#to remove
					cv2.circle(frame, lcentre, int(lrad),(0,255,0),1)
					cv2.circle(frame, rcentre, int(rrad),(0,255,0),1)

					pos, ir_ratio = Position(rcentre, meshc[R_RIGHT], meshc[R_LEFT][0])
					#print(ir_ratio)
					#to remove
					cv2.putText(frame, f'Position: {pos} {ir_ratio:0.2f}', (30,90), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 1)
					if blinkcount>2:
						totalblinks += 1
						blinkcount = 0

				#to remove
				cv2.putText(frame, f'Blinks: {totalblinks}', (30,70), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 1)
				


				#visualizing our iris
				'''
				(lx,ly), lrad = cv2.minEnclosingCircle(meshc[L_IRIS])
				(rx,ry), rrad = cv2.minEnclosingCircle(meshc[R_IRIS])
				#cv2.circle(frame, np.array([lx,ly], dtype = np.int32), int(lrad),(0,255,0),1)
				#cv2.circle(frame, np.array([rx,ry], dtype = np.int32), int(rrad),(0,255,0),1)
				#cv2.polylines(frame,[meshc[L_IRIS]],True,(0,255,0),1,cv2.LINE_AA)
				#cv2.polylines(frame,[meshc[R_IRIS]],True,(0,255,0),1,cv2.LINE_AA)

				lcentre = np.array([lx, ly], dtype=np.int32)
				rcentre = np.array([rx, ry], dtype=np.int32)

				#to remove
				cv2.circle(frame, lcentre, int(lrad),(0,255,0),1)
				cv2.circle(frame, rcentre, int(rrad),(0,255,0),1)

				pos, ir_ratio = Position(rcentre, meshc[R_RIGHT], meshc[R_LEFT][0])
				#print(ir_ratio)
				#to remove
				cv2.putText(frame, f'Position: {pos} {ir_ratio:0.2f}', (30,90), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 1)


				#to ''
				#Eye identification
				r = [meshc[p] for p in R_EYE]
				l = [meshc[p] for p in L_EYE]
				cropr, cropl = Eyes(frame,r,l)
				
				rposition, rcolor = Position(cropr)
				cv2.putText(frame, f'RightEye: {rposition}', (30,90), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), 1)
				lposition, lcolor = Position(cropl)
				cv2.putText(frame, f'LeftEye: {lposition}', (30,110), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), 1)

				print(results.multi_face_landmarks[0])
				for fl in results.multi_face_landmarks:
					#print(fl)
					draw.draw_landmarks(frame,landmark_list=fl, connections=mesh.FACEMESH_CONTOURS, landmark_drawing_spec=spec)
				#[print(f.x,f.y) for f in results.multi_face_landmarks[0].landmark]
				'''

			diff = time.time() - start
			fps = framecount/diff
			#to remove
			cv2.putText(frame, f'Frames: {fps: 0.0f}', (30,130), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 1)
			cv2.imshow('webcam', frame)
			
			key = cv2.waitKey(1)
			if key == ord('q'):
				break
	cam.release()
	cv2.destroyAllWindows()

'''
async def main():
	BehaviourAnalysis()
	await asyncio.sleep(1)
	FacialTracker()
  
asyncio.run(main())

'''  
if __name__=='__main__':
	import cProfile
	import pstats
	'''
	p1 = Process(target=BehaviourAnalysis)
	p1.start()
	p2 = Process(target=FacialTracker)
	p2.start()
	
	p1.join()
	p2.join()
	
	start = time.perf_counter()
	with Pool() as pool:
		pool.imap_unordered(BehaviourAnalysis)
		pool.imap_unordered(FacialTracker)
	end = time.perf_counter()
	print('Time taken: ', end-start)
	'''
	with cProfile.Profile() as cp:
		loop = asyncio.get_event_loop()
		loop.run_until_complete(asyncio.gather(BehaviourAnalysis(), FacialTracker()))
	stats = pstats.Stats(cp)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.print_stats()
