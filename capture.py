import cv2
import numpy as np 
from PIL import Image
from matplotlib import pyplot as plt
import scipy.misc

font = cv2.FONT_HERSHEY_SIMPLEX

def cropImage(frame,x,y,w,h):
    cropped_img = (frame[y:y+h,x:x+h])
    return cropped_img



def capt():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
    side_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    cap = cv2.VideoCapture(0)


    count = 0
    p = 1
    face = []
    side = []
    print('press \'q\' to exit...')
    while True:
        count+=1
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray,1.3,5)
        side = side_face_cascade.detectMultiScale(gray,1.3,5)
        li_cords = list(face_cascade.detectMultiScale(gray,1.3,5))
        cv2.putText(frame,'Press q to exit',(5,20),font,1,(0,0,0),1,cv2.LINE_AA)

        if len(face)>0:
            for(x,y,w,h) in face:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(150,255,190),2)
                if count%2==0:  
                    img = cropImage(frame,x,y,w,h)
                    cv2.imwrite('images\pos\img_{}.jpg'.format(int(count/2)),img)
        elif len(side)>0:
            for(x,y,w,h) in side:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(150,255,190),2)
                if count%2==0:  
                    img = cropImage(frame,x,y,w,h)
                    cv2.imwrite('images\pos\img_side_{}.jpg'.format(int(count/2)),img)

                

        cv2.imshow("frame",frame)   

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
