import cv2, os
import numpy as np
from PIL import Image

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer =  cv2.face.LBPHFaceRecognizer_create()
path='Face recogization/Database'
def check_file():
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    if os.path.exists('Face recogization/face_recognizer'):
        os.remove('Face recogization/face_recognizer')
check_file()   


def get_images_and_labels(path):
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for image_path in image_paths:
        
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[-1].split(".")[0].split(")")[0].split("(")[1].replace("Image", ""))  

        faces = faceCascade.detectMultiScale(image)
       
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(100)
    
    return images, labels



images, labels = get_images_and_labels(path)

recognizer.train(images, np.array(labels))
recognizer.save('Face recogization/face_recognizer.xml')
#recognizer.save('Face recogization/face_recognizer.xml')
cv2.destroyAllWindows()

