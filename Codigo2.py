import os
import cv2
import numpy as np

caminho_lfw = "C:\\Users\\Rodrigo Rossi\\Desktop\\testelfw\\treinamento" #caminho da base de treinamento LFW

nomes = os.listdir(caminho_lfw) #nomes das pessoas
nomes_auxiliar = []

def aplicaLBP (img):
    
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascata = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    rostos = cascata.detectMultiScale(cinza, scaleFactor=1.2, minNeighbors=5);
    if(len(rostos)==0):
        return None, None    
    (x, y, w, h) = rostos[0]    
    return cinza[y:y+w, x:x+h], rostos[0]

def prepararDados(caminho):
    diretorio = os.listdir(caminho) #nomes das pessoas
    rostos = []
    ids = []
    aux = 0
    for i in diretorio: #percorre cada pasta
        pasta_pessoa = caminho_lfw + "\\" + i   
        subdiretorio = os.listdir(pasta_pessoa)
        for j in subdiretorio: #percorre cada arquivo
            arquivo = pasta_pessoa + "\\" + j
            imagem = cv2.imread(arquivo)
            rosto, retangulo = aplicaLBP(imagem)
            if rosto is not None:
                rostos.append(rosto)
                nomes_auxiliar.append(i)
                ids.append(aux)
                aux=aux+1
    return rostos, ids

rostos, ids = prepararDados(caminho_lfw)

reconhecedor = cv2.face.LBPHFaceRecognizer_create()

reconhecedor.train(rostos, np.array(ids))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = aplicaLBP(img)

    label, confidence = reconhecedor.predict(face)
    label_text = nomes_auxiliar[label]
    
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

test_img1 = cv2.imread("C:\\Users\\Rodrigo Rossi\\Desktop\\testelfw\\teste\\Dick_Cheney_0003.jpg")
test_img2 = cv2.imread("C:\\Users\\Rodrigo Rossi\\Desktop\\testelfw\\teste\\George_W_Bush_0001.jpg")
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

cv2.imshow(nomes[0], predicted_img1)
cv2.imshow(nomes[1], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()