import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

path='Face recogization'

def check_file():
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    if not os.path.exists('Face recogization/Database'):
        os.makedirs('Face recogization/Database')
    if not os.path.exists('Face recogization/info.txt'):
        f=open('Face recogization/info.txt','w')
        f.close()
check_file()

with open('./Face recogization/info.txt','r+')as f:
        lines= list(f)
        pic_num= len(lines)
        if(pic_num==0):
            pic_num=1
        elif(pic_num>0):
            pic_num+=1
        
x=0

print(len(lines))    

while x==0:
    

    name='Surya'
    
    cap= cv2.VideoCapture(0)
    
    while True:
            ret,img = cap.read()
            
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray, (100, 100))
            cv2.imshow('img',img)
            
            k= cv2.waitKey(30) & 0xff
            if k ==27:
                x=1
                break
            if k==32:
                
                cv2.imwrite('Face recogization/Database/Image ('+str(pic_num)+').jpg',resized_image)
                with open('Face recogization/info.txt','a') as f:
                    
                    f.write(str(name)+"_"+str(pic_num)+'\n')
                pic_num+=1
            
            
            
            
                
    cap.release()

    cv2.destroyAllWindows()
