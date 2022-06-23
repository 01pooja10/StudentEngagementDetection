import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
import imutils
from imutils import face_utils
import warnings
warnings.filterwarnings('ignore')


def EAR(landmarks, frame):
	left = []
	right = []
	alarm = 0

	# left eye
	for i in range(36,42):
		x = landmarks.part(i).x
		y = landmarks.part(i).y
		left.append((x,y))

		nextpt1 = i + 1
		if i == 41:
			nextpt1 = 36

		x1 = landmarks.part(nextpt1).x
		y1 = landmarks.part(nextpt1).y

	# right eye
	for i in range(42,48):
		x = landmarks.part(i).x
		y = landmarks.part(i).y
		right.append((x,y))

		nextpt2 = i + 1
		if i == 47:
			nextpt2 = 42

		x1 = landmarks.part(nextpt2).x
		y1 = landmarks.part(nextpt2).y

	# calculate ratio
	left_EAR = aspect_ratio(left)
	right_EAR = aspect_ratio(right)
	ratio = ((left_EAR + right_EAR)/2)
	return ratio

def aspect_ratio(eye):
	eye = np.squeeze(eye)
	a = np.linalg.norm(eye[1] - eye[5])
	b = np.linalg.norm(eye[2] - eye[4])
	c = np.linalg.norm(eye[0] - eye[3])

	return (a + b)/(2.0 * c)

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

def detector():
	capture = cv2.VideoCapture(0)

	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor(r"C:\Users\Pooja\code\mitacs\imgs\shape_predictor_68_face_landmarks.dat")
	c = 0
	fps = 30
	skip = 10
	(lstart,lend) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
	(lstart,rend) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

	#alarm = 0
	#placeHolder2 = st.empty()

	while True:
		ret, frame = capture.read()

		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray,(128,128),interpolation=cv2.INTER_LINEAR)
		#cv2.imshow("Webcam", gray)
		#if fps%skip == 0:
		faces = detect(gray)

		for face in faces:
			p = face.left()
			q = face.top()
			r = face.right()
			s = face.bottom()
			landmarks = predict(gray, face)

			for i in range(68):
				x = landmarks.part(i).x
				y = landmarks.part(i).y

			yawn_threshold = 20
			eye_closed = 0.26
			eye_threshold = 30

			x = EAR(landmarks, gray)
			if x < eye_closed:
				c += 1
				if c >= eye_threshold:
					#song = AudioSegment.from_wav('file dependencies/alarm.wav')
					#play(song)
					cv2.putText(gray,"YOU'RE SLEEPING!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
					c = 0

			#y = yawn(landmarks, gray)

			for face in faces:
				p = face.left()
				q = face.top()
				r = face.right()
				s = face.bottom()
				landmarks = predict(gray, face)

				for i in range(68):
					x = landmarks.part(i).x
					y = landmarks.part(i).y

			y = yawn(landmarks, gray)
			if y > yawn_threshold:
				cv2.putText(gray,"YOU SEEM DROWSY!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

			cv2.imshow("Webcam", gray)

		key = cv2.waitKey(1)
		if key == 27:
			break

	capture.release()
	cv2.destroyAllWindows()

def detector2():
	capture = cv2.VideoCapture(0)

	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor(r"C:\Users\Pooja\code\mitacs\imgs\shape_predictor_68_face_landmarks.dat")
	c = 0
	fps = 30
	skip = 10
	eye_closed = 0.26
	eye_frames = 38
	yawn_threshold = 23

	(lstart,lend) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
	(rstart,rend) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
	(mstart,mend) = face_utils.FACIAL_LANDMARKS_IDXS['inner_mouth']

	while True:
		ret, frame = capture.read()

		#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#frame = cv2.resize(frame,(512,512),interpolation=cv2.INTER_LINEAR)
		#cv2.imshow("Webcam", gray)
		#if fps%skip == 0:
		faces = detect(frame,0)


		for face in faces:
			landmarks = predict(frame, face)
			landmarks = face_utils.shape_to_np(landmarks)

			l_eye = landmarks[lstart:lend]
			r_eye = landmarks[rstart:rend]
			mth = landmarks[mstart:mend]
			left_EAR = aspect_ratio(l_eye)
			right_EAR = aspect_ratio(r_eye)

			final = (left_EAR + right_EAR) / 2
			#y = yawn(landmarks, gray)

			left_hull = cv2.convexHull(l_eye)
			right_hull = cv2.convexHull(r_eye)
			mouth = cv2.convexHull(mth)
			cv2.drawContours(frame,[left_hull],-1,(0,255,0),1)
			cv2.drawContours(frame,[right_hull],-1,(0,255,0),1)
			cv2.drawContours(frame,[mouth],-1,(0,255,0),1)

			p = face.left()
			q = face.top()
			r = face.right()
			s = face.bottom()
			landmarks = predict(frame, face)

			y = yawn(landmarks,frame)
			if y > yawn_threshold:
				cv2.putText(frame,"YOU SEEM DROWSY!", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

			if final<eye_closed:
				c+=1
				if c>=eye_frames:
					cv2.putText(frame, 'DROWSY', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
					print('drowsy')
			else:
				c = 0
		cv2.imshow('Webcam feed',frame)

		key = cv2.waitKey(1)
		if key == 27:
			break

	capture.release()
	cv2.destroyAllWindows()


detector2()
