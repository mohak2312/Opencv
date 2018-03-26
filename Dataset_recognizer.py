import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Face recogization/face_recognizer.xml')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)

while True:   
    ret, im =cam.read()
    
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3,5)
    name=''
    
    
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        print(conf)
        f=open('Face recogization/info.txt')
        lines = f.read().splitlines()
        name=lines[Id-1].split("_")
        print(name)

        cv2.putText(im,str(name[0]),(x,y-3),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
       
        
    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF==ord('z'):
        break

cam.release()
cv2.destroyAllWindows()
