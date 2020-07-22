import cv2
import numpy as np
import sys

def getAreaOfFood(img1):
        #BGR에서 그레이 스케일이미지로 변환
	img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	#미디언 블러링으로 노이즈 제거
	img_filt = cv2.medianBlur( img, 5)
	#이미지 이진화
	img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# 접시와 음식에 해당하는 가장큰 윤곽선을 찾는다.
	mask = np.zeros(img.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	#검은색 배경에 가장큰 컨투어만 그림
	cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255), -1)
	#입력이미지에서 가장 큰 컨투어만 추출
	img_bigcontour = cv2.bitwise_and(img1,img1,mask = mask)

	#hsv채널로 변경
	hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv_img)
	#inRange함수로 plate부분의 영역을 설정
	mask_plate = cv2.inRange(hsv_img, np.array([0,0,100]), np.array([255,90,255]))
	mask_not_plate = cv2.bitwise_not(mask_plate)
	#가장 큰 컨투어(plate)에서 plate가 아닌 부분만(음식) 추출
	fruit_skin = cv2.bitwise_and(img_bigcontour,img_bigcontour,mask = mask_not_plate)

	#피부를 제외하기 위해 fruit skin을 hsv채널로 변환
	hsv_img = cv2.cvtColor(fruit_skin, cv2.COLOR_BGR2HSV)
	#피부 영역을 설정
	skin = cv2.inRange(hsv_img, np.array([0,10,60]), np.array([10,160,255])) #Scalar(0, 10, 60), Scalar(20, 150, 255)
	not_skin = cv2.bitwise_not(skin)
	#피부가 아닌 부분만 설정, 음식 픽셀만 획득
	fruit = cv2.bitwise_and(fruit_skin,fruit_skin,mask = not_skin)

	#그레이 이미지로 변환
	fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
	#흑백 바이너리 이미지
	fruit_bin = cv2.inRange(fruit_bw, 10, 255) #binary of fruit
	#윤곽선을 찾기전 3x3 타원 커널을 사용한 erosion으로 어두운 영역을 늘림
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	erode_fruit = cv2.erode(fruit_bin,kernel,iterations = 1)

	#음식이 될 가장큰 윤곽선을 찾는다
	#erosion된 음식부분의 이미지를 이진화
	img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	#음식만 추출하기위해 이진화된 이미지의 가장큰 윤곽선을 마스크로 설정
	cv2.drawContours(mask_fruit, [largest_areas[-2]], 0, (255,255,255), -1)
	#13x13 타원 커널을 사용한 dilation으로 마스크의 밝은 영역을 늘림
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
	mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
	#fruit_bin에서 음식부분만 추출
	res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit2)
	#img1에서 음식부분만 추출
	fruit_final = cv2.bitwise_and(img1,img1,mask = mask_fruit2)
	
	#최종 음식 마스크 area 면적계산
	img_th = cv2.adaptiveThreshold(mask_fruit2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	largest_areas = sorted(contours, key=cv2.contourArea)
	fruit_contour = largest_areas[-2]
	fruit_area = cv2.contourArea(fruit_contour)

	
	#finding the area of skin. find area of biggest contour
	#신체의 피부 영역과 가장큰 윤곽선의 area를 계산
	skin2 = skin - mask_fruit2

	#피부 영역의 윤곽선을 찾기전 erosion
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	skin_e = cv2.erode(skin2,kernel,iterations = 1)
	#피부영역의 마스크 계산
	img_th = cv2.adaptiveThreshold(skin_e,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_skin = np.zeros(skin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	cv2.drawContours(mask_skin, [largest_areas[-2]], 0, (255,255,255), -1)

	#피부의 가장큰 윤곽선에 외접하는 가장작은 직사각형을 계산
	skin_rect = cv2.minAreaRect(largest_areas[-2])
	#직사각형의 각 꼭지점 반환
	box = cv2.boxPoints(skin_rect)
	print(box)
	box = np.int0(box)
	mask_skin2 = np.zeros(skin.shape, np.uint8)
	cv2.drawContours(mask_skin2,[box],0,(255,255,255), -1)

	#경계사각형의 height 계산
	pix_height = max(skin_rect[1])
	# cm단위로 변환시키는 변수
	pix_to_cm_multiplier = 5.0/pix_height
	skin_area = cv2.contourArea(box)
	


	return fruit_area, mask_fruit2, fruit_final, skin_area, fruit_contour, pix_to_cm_multiplier


if __name__ == '__main__':
	img1 = cv2.imread("./images/All_Images/1_1.jpg")
	area, bin_fruit, img_fruit, skin_area, fruit_contour, pix_to_cm_multiplier = getAreaOfFood(img1)

	cv2.waitKey()
	cv2.destroyAllWindows()

