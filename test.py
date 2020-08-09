# -*- coding: utf-8 -*-

import numpy as np
import cv2
from create_feature import *
from calorie_calc import *
from learn import *
import csv

def output(path):

    # svm_model = cv2.ml.SVM_create()
    svm_model = cv2.ml.SVM_load('svm_data.dat')
    feature_mat = []
    response = []
    image_names = []
    pix_cm = []
    fruit_contours = []
    fruit_areas = []
    fruit_volumes = []
    fruit_mass = []
    fruit_calories = []
    skin_areas = []
    fruit_calories_100grams = []

    img_path = path
    print("이미지경로 : " , img_path)
    fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
    pix_cm.append(pix_to_cm)
    fruit_contours.append(fcont)
    fruit_areas.append(farea)
    feature_mat.append(fea)
    skin_areas.append(skinarea)
    image_names.append(img_path)

    testData = np.array(feature_mat, dtype=np.float32).reshape(-1, 94)
    result = svm_model.predict(testData)[1]
    result = np.ravel(result)

    # calculate calories
    volume = getVolume(result[0], fruit_areas[0], skin_areas[0], pix_cm[0], fruit_contours[0])
    mass, cal, cal_100 = getCalorie(result[0], volume)
    fruit_volumes.append(volume)
    fruit_calories.append(cal)
    fruit_calories_100grams.append(cal_100)
    fruit_mass.append(mass)

    print("예측레이블 : ", result[0])
    print("예측과일 : 나중에 if문 하면 나옴")
    print("예측부피 : ", mass)
    print("예측칼로리 : ", cal)
    print("100g당칼로리: ", cal_100)

if __name__ == '__main__':
    # 사용자가 이미지 업로드 하면 이미지파일 경로획득
    # 이미지파일의 해상도,크기조절 해야할듯?

    path = "C:/Calorie Estimation/images/All_Images/1_1.jpg"

    # path 획득후 output 돌리면
    # 출력해야할것 : 사용자가 입력한 이미지, 음식이름, 칼로리, 추가정보 등등 ?

    output(path)
