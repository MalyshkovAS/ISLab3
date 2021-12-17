#Импорт модулей необходимых для работы программы
import argparse
import cv2
import math
from dataclasses import dataclass
import time

smileCount = 0
lastTimeSmile = time.time()
smileToWaitParam = 3

argumentParser = argparse.ArgumentParser()
argumentParser.add_argument("-f", "--frontFace", required = True)
argumentParser.add_argument("-e", "--eye", required = True)
argumentParser.add_argument("-s", "--smile", required = True)
args = vars(argumentParser.parse_args())
faceDetection = cv2.CascadeClassifier(args["frontFace"]) 
eyeDectection = cv2.CascadeClassifier(args["eye"]) 
smileDetection = cv2.CascadeClassifier(args["smile"])

@dataclass
class NeuralFaceDetection :
    featureName : str
    param1 : float
    param2 : float

#Функция определения черт лица
#Возвращает тоже изображение с обведенными чертами лица
def detection(grayscale, img):
    #Определения лица
    detectedFace = faceDetection.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in detectedFace:
        #Обводим лицо
        cv2.rectangle(img,
                     (x_face, y_face),
                     (x_face + w_face, y_face + h_face),
                     (0, 43, 255), 2)
        faceParam = NeuralFaceDetection("face",x_face,y_face)
        ri_grayscale = grayscale[y_face:y_face + h_face, x_face:x_face + w_face]
        ri_color = img[y_face:y_face + h_face, x_face:x_face + w_face] 
        #Определение глаз
        detectedEye = eyeDectection.detectMultiScale(ri_grayscale, 1.2, 18) 
        for (x_eye, y_eye, w_eye, h_eye) in detectedEye:
            eyeParam = NeuralFaceDetection("eye",x_eye,y_eye)
            rectangleForEye1 = x_eye + w_eye
            rectangleForEye2 = y_eye + h_eye
            cv2.rectangle(ri_color,
                          (x_eye, y_eye),
                          (rectangleForEye1 ,rectangleForEye2),
                          (17, 255, 0), 2) 
        #Определение улыбки
        detectedSmile = smileDetection.detectMultiScale(ri_grayscale, 1.7, 20)
        for (x_smile, y_smile, w_smile, h_smile) in detectedSmile:
            global smileCount
            global lastTimeSmile
            smileParam = NeuralFaceDetection("smile", x_smile, y_smile)
            rectangleForSmile1 = x_smile + w_smile
            rectangleForSmile2 = y_smile + h_smile
            if time.time() - lastTimeSmile > smileToWaitParam:
                smileCount = smileCount+1
                print("Улыбнулся ", smileCount, " раз")
                lastTimeSmile = time.time()
            cv2.putText(ri_color,
                          "Smile detected",
                          (20 ,20),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          (17, 255, 0),
                          2,cv2.LINE_AA)
            cv2.rectangle(ri_color,
                          (x_smile, y_smile),
                          (rectangleForSmile1, rectangleForSmile2),
                          (255, 0, 72), 2)
    return img 
#Инициализация веб камеры
print("Инициализация...")
vc = cv2.VideoCapture(0) 
#Бесконечный цикл для получения потока видео с веб камеры
while True:
    #Реадер изображения
    _, img = vc.read() 
    #Задание цвета изображения
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Вызов функции определения черт лица
    final = detection(grayscale, img) 
    #Наложение результата определения лица на поток видео
    cv2.imshow('Video', final) 
    #Ожидание кода выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
#Остановка потока видео и закрытие окна
vc.release() 
cv2.destroyAllWindows()