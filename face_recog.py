from keras.models import load_model
import cv2 
import numpy as np 
import time

clf = load_model('face_recog.hdf5')



def cropImage(frame,x,y,w,h):
    cropped_img = (frame[y:y+h,x:x+h])
    return cropped_img


def frame_preprocess_pipeline(img):
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(256,256))
    img = np.array(img)
    img = img.astype('float')
    img /= 255
    img = np.expand_dims(img,axis=3)
    img = np.expand_dims(img,axis=0)
    return img




face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
side_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
cap = cv2.VideoCapture(0)


font = cv2.FONT_HERSHEY_SIMPLEX

face = []
side = []
print('press \'q\' to exit...')
while True:
   
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.3,5)
    side = side_face_cascade.detectMultiScale(gray,1.3,5)
    li_cords = list(face_cascade.detectMultiScale(gray,1.3,5))
    check_frame = cv2.resize(gray,(256,256)).flatten()
    cv2.putText(frame,'Press q to exit',(5,20),font,1,(0,0,0),1,cv2.LINE_AA)
    if len(face)>0:
        for(x,y,w,h) in face:           
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,190),2)    
            img = cropImage(frame,x,y,w,h)
            img = frame_preprocess_pipeline(img)
            if clf.predict_classes(img) == 0:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                print(clf.predict_classes(img))
                cv2.putText(frame,'There You are',(5,180),font,1,(255,0,0),2,cv2.LINE_AA)


    
    elif len(side)>0:
        for(x,y,w,h) in side:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,190),2)    
            img = cropImage(frame,x,y,w,h)
            img = frame_preprocess_pipeline(img)
            if clf.predict_classes(img) == 0:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                print(clf.predict_classes(img))
                cv2.putText(frame,'There You are',(5,180),font,1,(255,0,0),2,cv2.LINE_AA)
           


    cv2.imshow("frame",frame)   

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
