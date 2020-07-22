from feature_moments import getShapeFeatures
from feature_gabor import *
from feature_color import getColorFeature
from img_seg import *

def createFeature(img):
	'''
	이미지의 세 가지 특징(색상, 질감, 모양)을 사용하여 이미지의 특징 벡터 추출 함수 생성
	'''
	feature = []
	areaFruit, binaryImg, colourImg, areaSkin, fruitContour, pix_to_cm_multiplier = getAreaOfFood(img)
	#이미지 세그멘테이션에 이미지 값을 넣은 리턴 값을 각각 변수에 할당
	color = getColorFeature(colourImg)
	texture = getTextureFeature(colourImg)
	shape = getShapeFeatures(binaryImg)
	#세 가지 특징 추출 함수 입력 파라미터에 필요한 이미지 값을 넣은 후 리턴 값을 각각 변수에 할당

	for i in color:
		feature.append(i)
	for i in texture:
		feature.append(i)
	for i in shape:
		feature.append(i)
	#각 할당된 데이터를 feature에 추가
	M = max(feature)
	m = min(feature)
	feature = list(map(lambda x: x * 2, feature))
	feature = (feature - M - m)/(M - m)
	mean=np.mean(feature)
	dev=np.std(feature)
	feature = (feature - mean)/dev

	return feature, areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier
	#특징, 음식 면적, 손가락 면적, 음식 윤곽, ... 을 리턴한다.
def readFeatureImg(filename):
	'''
	파일이름이 주어질 때 입력 이미지를 읽어서
	이미지에 대한 특징 벡터 생성
	'''
	img = cv2.imread(filename)
	f, farea, skinarea, fcont, pix_to_cm = createFeature(img)
	return f, farea, skinarea, fcont, pix_to_cm
if __name__ == '__main__':
	import sys
	f = readFeatureImg(sys.argv[1])
	print (f, len(f))
