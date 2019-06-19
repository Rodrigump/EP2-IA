import cv2
import os

dir_imagem = "real-madrid.jpg"
dir_cascata = "haarcascade_frontalface_default.xml"

cascata = cv2.CascadeClassifier(dir_cascata)
        
imagem = cv2.imread(dir_imagem)
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
rostos = cascata.detectMultiScale(
        cinza,
        1.1,
        5,
        minSize = (30,30)
        )
    
print("Foram detectados {0} rostos!".format(len(rostos)))
    
for (x,y,w,h) in rostos:
    cv2.rectangle(imagem, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
cv2.imshow("Faces found", imagem)
cv2.waitKey(0)    