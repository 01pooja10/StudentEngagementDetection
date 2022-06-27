import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as npc

#constants
FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOW = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UP = [185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Left eyes indices
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
L_BROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
L_IRIS = [474, 475, 476, 477]

# right eyes indices
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
R_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
R_IRIS = [469, 470, 471, 472]


mesh = mp.solutions.face_mesh
draw = mp.solutions.drawing_utils
spec = draw.DrawingSpec(thickness=1,circle_radius=1)

cam = cv2.VideoCapture(0)

def Landmarks(img,results,draw=False):
    h, w = img.shape[:2]
    coords = np.array([(int(f.x * w),int(f.y * h)) for f in results.multi_face_landmarks[0].landmark])
    if draw:
        [cv2.circle(img,p,2,(0,255,0),-1) for p in coords]
    return coords


def Euclidean(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    sq1 = (x2-x1)**2
    sq2 = (y2-y1)**2
    euc = np.sqrt(sq1+sq2)
    return euc


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
    #cv2.imshow('mask', mask)

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
    #print(cropr, cropl)
    #cv2.imshow('r', cropr)
    #cv2.imshow('l', cropl)
    return cropr,cropl


bc = 0
tb = 0
#driver code
with mesh.FaceMesh(max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as fm:
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame.flags.writeable = False

        results = fm.process(frame)
        #frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            meshc = Landmarks(frame, results, False)

            ratio = Blink(frame, meshc, R_EYE, L_EYE)
            cv2.putText(frame,f'Ratio: {ratio}',(30,50),cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0),1)

            if ratio>5.5:
                bc=bc+1
                #cv2.putText(frame, 'Blink', (30,60), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0), 1)

            else:
                if bc>4:
                    tb+=1
                    bc = 0

            cv2.putText(frame, f'Blinks: {tb}', (30,70), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0), 1)

            #visualizing our iris

            (lx,ly), lrad = cv2.minEnclosingCircle(meshc[L_IRIS])
            (rx,ry), rrad = cv2.minEnclosingCircle(meshc[R_IRIS])
            cv2.circle(frame, np.array([lx,ly], dtype = np.int32), int(lrad),(0,255,0),1)
            cv2.circle(frame, np.array([rx,ry], dtype = np.int32), int(rrad),(0,255,0),1)
            #cv2.polylines(frame,[meshc[L_IRIS]],True,(0,255,0),1,cv2.LINE_AA)
            #cv2.polylines(frame,[meshc[R_IRIS]],True,(0,255,0),1,cv2.LINE_AA)

            #Eye identification
            r = [meshc[p] for p in R_EYE]
            l = [meshc[p] for p in L_EYE]
            cropr, cropl = Eyes(frame,r,l)
            

            '''
            print(results.multi_face_landmarks[0])
            for fl in results.multi_face_landmarks:
                #print(fl)
                draw.draw_landmarks(frame,landmark_list=fl, connections=mesh.FACEMESH_CONTOURS, landmark_drawing_spec=spec)
            #[print(f.x,f.y) for f in results.multi_face_landmarks[0].landmark]
            '''
        cv2.imshow('webcam', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()
