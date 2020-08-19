# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys

# density - gram / cm^3
density_dict = {0: 0.5, 1: 0.609, 2: 0.94, 3: 0.577, 4: 0.641, 5: 1.151, 6: 0.482, 7: 0.513,
                8: 0.641, 9: 0.481, 10: 0.641, 11: 0.521, 12: 0.881, 13: 0.228, 14: 0.650,
                15: 0.310, 16: 1.162, 17: 0.961, 18: 0.952, 19: 1.005, 20: 0.178}
# kcal
calorie_dict = {0: 50, 1: 52, 2: 88, 3: 347, 4: 41, 5: 402, 6: 47, 7: 39,
                8: 131, 9: 17, 10: 16, 11: 218, 12: 60, 13: 39, 14: 30,
                15: 452, 16: 155, 17: 28, 18: 57, 19: 46, 20: 66}

# carbohydrate
carbohydrate_dict = {0: 5, 1: 14, 2: 23, 3: 63, 4: 10, 5: 1.3, 6: 12, 7: 9,
                     8: 25, 9: 3.9, 10: 3.6, 11: 6, 12: 15, 13: 9, 14: 8,
                     15: 51, 16: 1.1, 17: 9, 18: 15, 19: 11.4, 20: 12.65}

# protein
protein_dict = {0: 0.5, 1: 0.3, 2: 1.1, 3: 21, 4: 0.9, 5: 25, 6: 0.9, 7: 1.1,
                8: 5, 9: 0.9, 10: 0.7, 11: 15, 12: 1.1, 13: 1.9, 14: 0.6,
                15: 4.9, 16: 13, 17: 1.1, 18: 0.4, 19: 0.7, 20: 1.91}

# fat
fat_dict = {0: 0.5, 1: 0.2, 2: 0.3, 3: 1.2, 4: 0.2, 5: 33, 6: 0.1, 7: 0.1,
            8: 1.1, 9: 0.2, 10: 0.11, 11: 15, 12: 0.5, 13: 0.4, 14: 0.2,
            15: 25, 16: 11, 17: 0.3, 18: 0.1, 19: 0.3, 20: 0.82}

# cholesterol
cholesterol_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 105, 6: 0, 7: 0,
                    8: 33, 9: 0, 10: 0, 11: 2, 12: 0, 13: 0, 14: 0,
                    15: 19, 16: 373, 17: 0, 18: 0, 19: 0, 20: 0}

# Natrium
natrium_dict = {0: 1, 1: 1, 2: 1, 3: 12, 4: 69, 5: 621, 6: 0, 7: 4,
                8: 6, 9: 5, 10: 0, 11: 532, 12: 3, 13: 9, 14: 1,
                15: 326, 16: 124, 17: 2, 18: 1, 19: 2, 20: 170}

# skin of photo to real multiplier
skin_multiplier = 5 * 2.3

def getCalorie(label, volume):  # volume in cm^3
    '''
    Inputs are the volume of the foot item and the label of the food item
    so that the food item can be identified uniquely.
    The calorie content in the given volume of the food item is calculated.
    '''
    calorie_100 = calorie_dict[int(label)]
    carbohydrate_100 = carbohydrate_dict[int(label)]
    protein_100 = protein_dict[int(label)]
    fat_100 = fat_dict[int(label)]
    cholesterol_100 = cholesterol_dict[int(label)]
    natrium_100 = natrium_dict[int(label)]

    if (volume == None):
        return None, None, calorie_100, carbohydrate_100, protein_100, fat_100, cholesterol_100, natrium_100
    density = density_dict[int(label)]
    mass = volume * density * 1.0
    calorie_tot = (calorie_100 / 100.0) * mass

    mass = round(mass, 2)
    calorie_tot = round(calorie_tot, 2)

    # calories and nutrients per 100 grams
    return mass, calorie_tot, calorie_100, carbohydrate_100, protein_100, fat_100, cholesterol_100, natrium_100


def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
    '''
    Using callibration techniques, the volume of the food item is calculated using the
    area and contour of the foot item by comparing the food item to standard geometric shapes
    '''

    area_fruit = (area / skin_area) * skin_multiplier  # area in cm^2

    label = int(label)
    volume = 100

    # sphere-apple,tomato,orange,kiwi,onion
    if label == 1 or label == 9 or label == 7 or label == 6 or label == 12 or label == 18 or label == 19:
        radius = np.sqrt(area_fruit / np.pi)
        volume = (4 / 3) * np.pi * radius * radius * radius

    # cylinder like banana, cucumber, carrot
    if label == 2 or label == 10 or (label == 4 and area_fruit > 30):
        fruit_rect = cv2.minAreaRect(fruit_contour)
        height = max(fruit_rect[1]) * pix_to_cm_multiplier
        radius = area_fruit / (2.0 * height)
        volume = np.pi * radius * radius * height
    # cheese, carrot, sauce, bread
    if (label == 4 and area_fruit < 30) or (label == 5) or (label == 11) or (label == 15) or (label == 20):
        volume = area_fruit * 0.5  # assuming width = 0.5 cm

    # pasta, watermelon, beans, capsicum
    if (label == 8) or (label == 14) or (label == 3) or (label == 13):
        volume = None

    # egg, lemon
    if (label == 16) or (label == 17):
        fruit_rect = cv2.minAreaRect(fruit_contour)
        height = max(fruit_rect[1]) * pix_to_cm_multiplier
        radius = area_fruit / (2.0 * height)
        volume = np.pi * radius * radius * height * 0.6

    return volume
