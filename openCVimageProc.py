import sys
import cv2
import numpy as np
import pickle

imagePath   = "D:\\code\\image_recognition\\imgTest\\test11.jpg"
faceCascade = cv2.CascadeClassifier("face_recognition_image\\FaceDetect\\haarcascade_frontalface_default.xml") 
image = cv2.imread(imagePath)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("D:\\code\\image_recognition\\database\\trainner.yml")
labels = {"person_name": 1}

with open("D:\\code\\image_recognition\\database\\labels.pickle",'rb') as f:

    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

for(x,y,w,h) in faces:
    roi_gray = gray[y:y+h,x:x+w]
    id_,conf  = recognizer.predict(roi_gray)
    print(id_)
    print(conf)
    if conf >=4 and conf <=170:

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        name = labels[id_]
        color = (255,255,255)
        stroke = 2
        cv2.putText(image,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

    color = (255,0,0)
    stroke = 2
    end_cord_x = x + w
    end_cord_y = y + h

    cv2.rectangle(image,(x,y),(end_cord_x,end_cord_y),color,stroke)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#D:\code\image_recognition\database\labels.pickle
#D:\code\image_recognition\database\trainner.yml
