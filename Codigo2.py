import os
import cv2
import numpy as np

caminho_lfw = "treinamento" #caminho da base de treinamento LFW

nomes = os.listdir(caminho_lfw) #nomes das pessoas
nomes_auxiliar = []

def aplicaLBP (img):
    
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascata = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    rostos = cascata.detectMultiScale(cinza, scaleFactor=1.2, minNeighbors=5)

    if(len(rostos)==0):
        return None, None    
    (x, y, w, h) = rostos[0]    
    return cinza[y:y+w, x:x+h], rostos[0]

def retornaLista(subpasta):
    # retorna lista com dados de arquivo .txt disponibilizado no caminho do parâmetro
    # deve esar em uma subpasta dentro da pasta raiz

    # monta caminho
    filename = os.path.abspath(os.curdir) + subpasta
    # Le arquivo e transforma em lista de listas
    with open(filename) as file:
        lista = [line.split() for line in file.read().splitlines()]

    return lista

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

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img, nome):
    img = test_img.copy()
    face, rect = aplicaLBP(img)

    label, confidence = reconhecedor.predict(face)
    label_text = nomes_auxiliar[label]

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img

def rodaTeste(dirTeste,nomes):

    indice = 0
    for t in dirTeste:
        dirImg = os.listdir("teste\\" + t)
        for img in dirImg:
            imagem_teste = cv2.imread("teste\\" + t + "\\" + str(img))
            preditor = predict(imagem_teste, t)
    cv2.imshow(nomes[indice], preditor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    indice = indice + 1

# prepara dados
rostos, ids = prepararDados(caminho_lfw)

#carrega arquivos texto da base LFW
lfw_names     = retornaLista("/txt_lfw/lfw-names.txt")
pairsDevTrain = retornaLista("/txt_lfw/pairsDevTrain.txt")
pairsDevTest  = retornaLista("/txt_lfw/pairsDevTest.txt")
pairs         = retornaLista("/txt_lfw/pairs.txt")

# cria reconhecedor LBP
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
# treina reconhecedor com as imagens da pasta treinamento
reconhecedor.train(rostos, np.array(ids))

# testes
indice = 0
testes = os.listdir("teste") #caminho do conjunto de testes
for t in testes:
    imagem_teste = cv2.imread("teste\\" + str(t))
    preditor = predict(imagem_teste,t)
    cv2.imshow(nomes[indice], preditor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    indice = indice+1

rodaTeste(testes, nomes)
