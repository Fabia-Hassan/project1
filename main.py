[20:15, 3/17/2022] Laiba crime Part: #Import Libraries
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2 as cv
import os

#Read or Load data
img=cv.imread('img1.jpg')
plt.imshow(img[:,:,::-1])

result=DeepFace.analyze(img)

plt.imshow(img[:,:,::-1])
plt.title("2k19-BSCS-321" + "\n" "Hello" +result["gender"])

#image Analyzer
print("Emotions:" ,result)
print("Angry:",+result['emotion']['angry'])
print("Disgust:",+result['emotion']['disgust'])
print("Fear:",+result['emotion']['fear'])
print("Happy:",+result['emotion']['happy'])
print("Sad:",+result['emotion']['sad'])
print("Surprise:",+result['emotion']['surprise'])
print("Neutral:",+result['emotion']['neutral'])
print("Age:",+result['age'])

plt.show()
[20:15, 3/17/2022] Laiba crime Part: #Import Libraries
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2 as cv
import os

#Read or Load data
img=cv.imread('im2.jpeg')
plt.imshow(img[:,:,::-1])

result=DeepFace.analyze(img)

plt.imshow(img[:,:,::-1])
plt.title("Hello" +result["gender"])

#image Analyzer
print("Emotions:" ,result)
print("Angry:",+result['emotion']['angry'])
print("Disgust:",+result['emotion']['disgust'])
print("Fear:",+result['emotion']['fear'])
print("Happy:",+result['emotion']['happy'])
print("Sad:",+result['emotion']['sad'])
print("Surprise:",+result['emotion']['surprise'])
print("Neutral:",+result['emotion']['neutral'])
print("Age:",+result['age'])

plt.show()
[20:15, 3/17/2022] Laiba crime Part: import cv2

#load the cascade
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#to capture video from webcam
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

#to use a video file as input
#cap=cv2.VideoCapture("filename.mp4")

while True:
   #read the frame
   ret,img=cap.read()
   #convert to grayscale
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   #detect the faces
   faces = face_cascade.detectMultiScale(gray,1.1,4)
   #draw the rectangle around each face
   for(x,y,w,h) in faces:
       cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
   #display
   cv2.imshow('img',img)
   #stop if escape key is pressed
   k=cv2.waitKey(30) & 0xff
   if k==27:
      break
#release the video capture
cap.release()