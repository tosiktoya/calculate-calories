# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys

# 이미지에서 형태 특징 추출
def getShapeFeatures(img):
	'''
	이미지의 shape feature 는 이미지의 contour 기반으로 Hu Moments 를 사용하여 구함
	'''

	# contours : 검출한 컨투어 좌표
	# hierarchy : 컨투어 계층정보
	# cv2.RETR_LIST : 모드, 모든라인을 계층없이 제공
	# cv2.CHAIN_APPROX_SIMPLE : method, 컨투어 꼭짓점 좌표만 제공

	contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# 이미지 모멘트
	# cv2.moments, cv2.HuMoments 이용
	# contour : 모멘트계산대상 컨투어좌표, moment : 결과 모멘트

	moments = cv2.moments(contours[0])
	hu = cv2.HuMoments(moments)
	feature = []
	for i in hu:
		feature.append(i[0])

	# 정규화
	M = max(feature)
	m = min(feature)
	feature = list(map(lambda x: x * 2, feature))
	feature = (feature - M - m)/(M - m)
	mean = np.mean(feature)
	dev = np.std(feature)
	feature = (feature - mean)/dev

	return feature

if __name__ == '__main__':
	#img = cv2.imread(sys.argv[1])
	img = cv2.imread("./images/All_Images/1_1.jpg", cv2.IMREAD_COLOR)
	img_contour = img.copy()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = cv2.inRange(img, 80, 255)
	img1 = cv2.bitwise_and(img, img, mask = mask)
	# shape feature vector 출력 testing
	h = getShapeFeatures(img1)
	print(h)
	cv2.waitKey()
	cv2.destroyAllWindows()