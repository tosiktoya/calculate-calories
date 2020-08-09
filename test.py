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

    for j in [1]:
        for i in range(21, 22):
            img_path = path
            print("이미지경로 : " , img_path)
            fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
            pix_cm.append(pix_to_cm)
            fruit_contours.append(fcont)
            fruit_areas.append(farea)
            feature_mat.append(fea)
            skin_areas.append(skinarea)
            response.append([float(j)])
            image_names.append(img_path)

    testData = np.array(feature_mat, dtype=np.float32).reshape(-1, 94)
    responses = np.array(response, dtype=int)
    result = svm_model.predict(testData)[1]
    result = np.ravel(result)
    mask = result == responses
    #print(result)

    # calculate calories
    for i in range(0, len(result)):
        #print(result[i])
        volume = getVolume(result[i], fruit_areas[i], skin_areas[i], pix_cm[i], fruit_contours[i])
        mass, cal, cal_100 = getCalorie(result[i], volume)
        fruit_volumes.append(volume)
        fruit_calories.append(cal)
        fruit_calories_100grams.append(cal_100)
        fruit_mass.append(mass)

        #print(' i and result', i, result)

    for i in range(0, len(mask)):
        if mask[i][0] == False:
            print( result[i], image_names[i])

    correct = np.count_nonzero(mask)
    #print(correct * 100.0 / result.size)

if __name__ == '__main__':
    # 경로설정
    #path = './images/Test_Images/1_21.jpg'
    path = "C:/Calorie Estimation/images/Apple/1_1.jpg"

    output(path)