import tensorflow as tf 
classifierLoad = tf.keras.models.load_model('model_v8.h5')

import numpy as np
import cv2
from keras.preprocessing import image

for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    for i in range(1, 26):
        test_data = "./images/All_Images/" + str(j) + "_" + str(i) + ".jpg"

        test_image = image.load_img(test_data, target_size=(200, 200))
        test_image1 = image.img_to_array(test_image)
        test_image2 = np.expand_dims(test_image1, axis=0)

        img = cv2.imread(test_data)
        img1 = cv2.resize(img, (400, 400))
        # img2 = np.expand_dims(img1, axis=0)

        model_out = classifierLoad.predict(test_image2)

        result = np.argmax(model_out)
        classes = {
            0: 'Apple',
            1: 'Banana',
            2: 'Beans',
            3: 'Carrot',
            4: 'Cheese',
            5: 'Orange',
            6: 'Onion',
            7: 'Pasta',
            8: 'Tomato',
            9: 'Cucumber',
            10: 'Sauce',
            11: 'Kiwi',
            12: 'Capsicum',
            13: 'Watermelon',
            14 : 'Doughnut',
            15 : 'Egg',
            16 : 'Lemon',
            17 : 'Pear',
            18 : 'Plum',
            19 : 'Bread'
        }
        name = classes[result]
        print(test_data, " label:", result+1, " name:", name)

