# -*- coding: utf-8 -*-
import cv2
import math
import sys
import numpy as np

# 이미지에서 색상 특징 추출
def getColorFeature(img):
    # BGR to HSV 변환
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv) # 채널별로 분리

    hsvHist = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(6)]

    featurevec = []

    # 이미지 히스토그램 생성
    # H : 0~180, S : 0~255, V : 0~255
    # cv2.calcHist(img, channel, mask, histSize, ranges)

    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [6, 2, 2], [0, 180, 0, 256, 0, 256])
    for i in range(6):
        for j in range(2):
            for k in range(2):
                featurevec.append(hist[i][j][k])

    feature = featurevec[1:]

    # 정규화
    M = max(feature)
    m = min(feature)
    feature = list(map(lambda x: x * 2, feature))
    feature = (feature - M - m) / (M - m)
    mean = np.mean(feature)
    dev = np.std(feature)
    feature = (feature - mean) / dev

    return feature

if __name__ == '__main__':
    #img = cv2.imread(sys.argv[1])
    img = cv2.imread("./images/All_Images/1_1.jpg", cv2.IMREAD_COLOR)
    featureVector = getColorFeature(img)
    #print(featureVector)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
