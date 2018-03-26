import cv2
import numpy as np
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time

cert_path = "c:/certs/"
host = "a2kc9la4cp40qj.iot.us-east-1.amazonaws.com"
returntopic = "$aws/things/ir2_proj/shadow/return"
sendtopic = "$aws/things/ir2_proj/shadow/up"
logtopic = "$aws/things/ir2_proj/log"
root_cert = cert_path + "root-CA.crt"
cert_file = cert_path + "ir2_proj.cert.pem"
key_file = cert_path + "ir2_proj.private.key"

def customCallback(client, userdata, message):
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")

#starting service
robot = AWSIoTMQTTClient(host)
robot.configureEndpoint(host, 8883)
robot.configureCredentials(root_cert, key_file, cert_file)

# AWSIoTMQTTClient connection configuration
robot.configureAutoReconnectBackoffTime(1, 32, 20)
robot.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
robot.configureDrainingFrequency(2)  # Draining: 2 Hz
robot.configureConnectDisconnectTimeout(10)  # 10 sec
robot.configureMQTTOperationTimeout(5)  # 5 sec

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Face recogization/face_recognizer.xml')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(1)


robot.connect()
a=0
last_name=''
last_frame=''
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
        a+=1
        print(a)
        if(a==50 ):
            if(name[0]!=last_name and name[0] == last_frame ):
                #robot.publish(sendtopic, str(name[0]),1)
                print('send')
                last_name=name[0]

            a=0
        

            
        last_frame=name[0]  
        cv2.putText(im,str(name[0]),(x,y-3),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
       
        
    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF==ord('z'):
        break

robot.disconnect()
cam.release()
cv2.destroyAllWindows()
