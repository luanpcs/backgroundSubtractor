import cv2 as cv
import numpy as np

def limiar(amostra):
    out = amostra.copy()
    out[amostra >  25] = 255
    out[amostra <= 25]  = 0
    return out

def BW (amostra):
    return ((0.2126 * amostra[:, :, 0]) + (0.7152 * amostra[:, :, 1]) + 
            (0.0722 * amostra[:, :, 2])).astype(np.uint8)

def exit():
    video.release()
    cv.destroyAllWindows()

video = cv.VideoCapture("videoRua.mp4")

_, firstFrame = video.read()
firstGray = BW(firstFrame)
firstGray = cv.GaussianBlur(firstGray, (5, 5), 0)

while True:
    _, frame = video.read()
    grayFrame = BW(frame)
    grayFrame = cv.GaussianBlur(grayFrame, (5, 5), 0)

    dif = cv.absdiff(firstGray, grayFrame)
    dif = limiar(dif)
    cv.imshow("Primeiro frame", firstFrame)
    cv.imshow("Frame atual", frame)
    cv.imshow("Subtracao", dif)

    if cv.waitKey(30) == 27:
      exit()