import os
import cv2
import numpy as np
from flask import Flask, request, render_template, flash, session, redirect
from werkzeug.utils import secure_filename
from create_feature import *
from calorie_calc import *
import tensorflow as tf
from keras.preprocessing import image
import csv

# 탄수화물, 단백질, 지방, 콜레스테롤, 나트륨 일일 권장섭취량, 칼로리 한끼 권장섭취량
nutrient_dict = {0: 130, 1: 100, 2: 51, 3: 300, 4: 2000, 5: 800}

# 100g당 칼로리
calorie_dict = {0: 50, 1: 52, 2: 88, 3: 347, 4: 41, 5: 402, 6: 47, 7: 39,
                8: 131, 9: 17, 10: 16, 11: 218, 12: 60, 13: 39, 14: 30,
                15: 452, 16: 155, 17: 28, 18: 57, 19: 46, 20: 66}

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#메인 페이지
@app.route('/')
def render_file():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            classes = {
                1 : 'Apple',
                2 : 'Banana',
                3 : 'Beans',
                4 : 'Carrot',
                5 : 'Cheese',
                6 : 'Orange',
                7 : 'Onion',
                8 : 'Pasta',
                9 : 'Tomato',
                10 : 'Cucumber',
                11 : 'Sauce',
                12 : 'Kiwi',
                13 : 'Capsicum',
                14 : 'Watermelon',
                15 : 'Doughnut',
                16 : 'Egg',
                17 : 'Lemon',
                18 : 'Pear',
                19 : 'Plum',
                20 : 'Bread'
                }

            image_name = f.filename
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'static', secure_filename(f.filename))
            f.save(file_path)
            
            svm_model = cv2.ml.SVM_load('svm_data.dat')
            pix_cm = []
            fruit_contours = []
            fruit_areas = []
            skin_areas = []
            feature_mat = []

            # 이미지에서 손가락이 있으면 True 상태 없으면 False 상태
            SkinState = True

            try:
                fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg("./static/"+f.filename)
                pix_cm.append(pix_to_cm)
                fruit_contours.append(fcont)
                fruit_areas.append(farea)
                feature_mat.append(fea)
                skin_areas.append(skinarea)

                # predict using svm model
                testData = np.array(feature_mat, dtype=np.float32).reshape(-1, 94)
                result_svm = svm_model.predict(testData)[1]
                result_svm = np.ravel(result_svm)

                # predict using cnn model
                classifierLoad = tf.keras.models.load_model('model_v8.h5')

                test_data = "./static/" + f.filename

                test_image = image.load_img(test_data, target_size=(200, 200))
                test_image1 = image.img_to_array(test_image)
                test_image2 = np.expand_dims(test_image1, axis=0)

                model_out = classifierLoad.predict(test_image2)
                result_cnn = np.argmax(model_out)

                if (result_svm[0] == (result_cnn + 1)):
                    result = result_svm[0]
                else:
                    result = result_cnn + 1

            # 이미지에서 손가락이 없으면 cnn 으로만 label 예측
            except cv2.error as e:
                SkinState = False

                # predict using cnn model
                classifierLoad = tf.keras.models.load_model('nonskincnn.h5')

                test_data = "./static/" + f.filename

                test_image = image.load_img(test_data, target_size=(200, 200))
                test_image1 = image.img_to_array(test_image)
                test_image2 = np.expand_dims(test_image1, axis=0)

                model_out = classifierLoad.predict(test_image2)
                result_cnn = np.argmax(model_out)

                result = result_cnn + 1

            except IndexError:
                SkinState = False

                # predict using cnn model
                classifierLoad = tf.keras.models.load_model('nonskincnn.h5')

                test_data = "./static/" + f.filename

                test_image = image.load_img(test_data, target_size=(200, 200))
                test_image1 = image.img_to_array(test_image)
                test_image2 = np.expand_dims(test_image1, axis=0)

                model_out = classifierLoad.predict(test_image2)
                result_cnn = np.argmax(model_out)

                result = result_cnn + 1

            result = np.ravel(result)

            # calculate calories
            # 손가락이 없을 경우에는 100g당 칼로리만 계산
            if(SkinState == True):
                volume = getVolume(result[0], fruit_areas[0], skin_areas[0], pix_cm[0], fruit_contours[0])
                mass, cal, cal_100, carbo, pro, fat, choles, nat = getCalorie(result[0], volume)
            elif(SkinState == False):
                volume = 0


                mass, cal, carbo, pro, fat, choles, nat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                cal_100 = round(calorie_dict[int(result)], 2)

            # Alarm message when exceeding appropriate intake volume
            Alarm_message = " "
            exceed_message = "이(가) 일일 적정섭취량을 초과하였습니다."

            if(carbo > nutrient_dict[0]):
                if(len(Alarm_message) < 3):
                    Alarm_message = Alarm_message + " 탄수화물 "
                else:
                    Alarm_message = Alarm_message + " , 탄수화물 "
            if(pro > nutrient_dict[1]):
                if (len(Alarm_message) < 3):
                    Alarm_message = Alarm_message + " 단백질 "
                else:
                    Alarm_message = Alarm_message + " , 단백질 "
            if(fat > nutrient_dict[2]):
                if (len(Alarm_message) < 3):
                    Alarm_message = Alarm_message + " 지방 "
                else:
                    Alarm_message = Alarm_message + " , 지방 "
            if(choles > nutrient_dict[3]):
                if (len(Alarm_message) < 3):
                    Alarm_message = Alarm_message + " 콜레스테롤 "
                else:
                    Alarm_message = Alarm_message + " , 콜레스테롤 "
            if(nat > nutrient_dict[4]):
                if (len(Alarm_message) < 3):
                    Alarm_message = Alarm_message + " 나트륨 "
                else:
                    Alarm_message = Alarm_message + " , 나트륨 "
            if(cal > nutrient_dict[5]):
                if(len(Alarm_message) < 3):
                    Alarm_message = Alarm_message + " 한끼칼로리 "
                else:
                    Alarm_message = Alarm_message + " , 한끼칼로리 "

            if(len(Alarm_message) > 3):
                message = Alarm_message + exceed_message
            else:
                message = " "

            class_name = classes[result[0]]

            # 다운로드용 텍스트 파일생성
            file = open('./static/txt/'+f.filename+'CalorieData.txt', 'w')
            file.write('음식명 : %s \n' %(class_name))
            file.write('질량 : %0.2f g\n' %(mass))
            file.write('칼로리 : %0.2f kcal\n' %(cal))
            file.write('100g당 칼로리 : %0.2f kcal\n' %(cal_100))
            file.write('탄수화물 : %0.2f g\n' %(carbo))
            file.write('단백질 : %0.2f g\n' %(pro))
            file.write('지방 : %0.2f g\n' %(fat))
            file.write('콜레스트롤 : %0.2f mg\n' %(choles))
            file.write('나트륨 : %0.2f mg\n' %(nat))
            file.close()

            download_path = './static/txt/'+f.filename+'CalorieData.txt'

            return render_template('upload.html', label=class_name, img=image_name, mass=mass, cal=cal,
                                   cal_100=cal_100, carbo=carbo, pro=pro, fat=fat,
                                   choles=choles, nat=nat, mes=message, download_path=download_path)
    return

if __name__ == '__main__':
    app.run(debug=True)
