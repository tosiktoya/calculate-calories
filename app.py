import os
import cv2
import numpy as np
from flask import Flask, request, render_template, flash, session, redirect
from werkzeug.utils import secure_filename
from create_feature import *
from calorie_calc import *
import csv

# 탄수화물, 단백질, 지방, 콜레스테롤, 나트륨
nutrient_dict = {0: 130, 1: 100, 2: 51, 3: 300, 4: 2000, 5: 800}

#from tensorflow import keras

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
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

            fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg("./static/"+f.filename)
            pix_cm.append(pix_to_cm)
            fruit_contours.append(fcont)
            fruit_areas.append(farea)
            feature_mat.append(fea)
            skin_areas.append(skinarea)

            testData = np.array(feature_mat, dtype=np.float32).reshape(-1, 94)
            result = svm_model.predict(testData)[1]
            result = np.ravel(result)

            # calculate calories
            volume = getVolume(result[0], fruit_areas[0], skin_areas[0], pix_cm[0], fruit_contours[0])
            mass, cal, cal_100, carbo, pro, fat, choles, nat = getCalorie(result[0], volume)

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
            return render_template('upload.html', label=class_name, img=image_name, mass=mass, cal=cal, cal_100=cal_100,
                                   carbo=carbo, pro=pro, fat=fat, choles=choles, nat=nat, mes=message)
    return

if __name__ == '__main__':
    app.run(debug=True)
