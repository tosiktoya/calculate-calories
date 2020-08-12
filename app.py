import os
import cv2
import numpy as np
from flask import Flask, request, render_template, flash, session, redirect
from werkzeug.utils import secure_filename
from create_feature import *
from calorie_calc import *
import csv

#from tensorflow import keras

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
            fruit_volumes = []
            fruit_mass = []
            fruit_calories = []
            fruit_calories_100grams = []
            skin_areas = []
            feature_mat = []
            #image_names = []

            fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg("./static/"+f.filename)
            pix_cm.append(pix_to_cm)
            fruit_contours.append(fcont)
            fruit_areas.append(farea)
            feature_mat.append(fea)
            skin_areas.append(skinarea)
            #image_names.append(img_path)
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

            #image = process_image(file_path)
            class_name = classes[result[0]]
            return render_template('upload.html', label = class_name, img = image_name, mass = mass, cal = cal, cal_100 = cal_100)
    return

if __name__ == '__main__':
    app.run(debug=True)
