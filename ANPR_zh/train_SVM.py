# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:26:32 2016

@author: tallt
"""
RESIZED_IMAGE_WIDTH = 136
RESIZED_IMAGE_HEIGHT = 36

import numpy as np
from numpy import *
import cv2
import os
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure

def feature_extraction(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    return hog_image
    
def getPlates():
    img_path = os.getcwd() + '\\resources\\svm\\has\\train\\'
    training_images = os.listdir(img_path)
    image_list = []
    label_list = []
    for image_name in training_images:
        img = cv2.imread(img_path + image_name)
        image_list.append(img)
        label_list.append(1)
    return image_list, label_list

def get_plate_test():
    img_path = os.getcwd() + '\\resources\\svm\\has\\test\\'
    training_images = os.listdir(img_path)
    image_list = []
    label_list = []
    for image_name in training_images:
        img = cv2.imread(img_path + image_name)
        image_list.append(img)
        label_list.append(1)
    return image_list, label_list
    
def getNoPlate():
    img_path = os.getcwd() + '\\resources\\svm\\no\\train\\'
    training_images = os.listdir(img_path)
    image_list = []
    label_list = []
    for image_name in training_images:
        img = cv2.imread(img_path + image_name)
        image_list.append(img)
        label_list.append(0)
    return image_list, label_list

def get_no_plate_test():
    img_path = os.getcwd() + '\\resources\\svm\\no\\test\\'
    training_images = os.listdir(img_path)
    image_list = []
    label_list = []
    for image_name in training_images:
        img = cv2.imread(img_path + image_name)
        image_list.append(img)
        label_list.append(0)
    return image_list, label_list

def get_test_data():
    plate_images, plate_labels = get_plate_test()
    no_images, no_labels = get_no_plate_test()
    for no_image in no_images:
        plate_images.append(no_image)
        plate_labels.append(0)    
    return plate_images, plate_labels
   
    
def get_train_data():
    plate_images, plate_labels = getPlates()
    no_images, no_labels = getNoPlate()
    for no_image in no_images:
        plate_images.append(no_image)
        plate_labels.append(0)    
    return plate_images, plate_labels
    
def flatten_images(images):
    flatten_images = []
    for image in images:
        image_hog = feature_extraction(image)
        flatten_img = reshape(image_hog, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)
        flatten_images.append(flatten_img)
    return flatten_images

def flatten_one_image(image):
    image_hog = feature_extraction(image)
    flatten_img = reshape(image_hog, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)
    return flatten_img
  
def predict_plate(image):
    svm_model = joblib.load('.//model//svm_model.m')
    image = flatten_one_image(image)
    predication = svm_model.predict([image])
    return predication
    if predication[0] == 0:
        return False
    elif predication[0] == 1:
        return True
    
def train():
    images, labels = get_train_data()
    flt_images = flatten_images(images)
    svm_model = svm_classifier(flt_images, labels)
    joblib.dump(svm_model, './/model//svm_model.m')
    print('training completed')
    return svm_model

def test():
    images, labels = get_test_data()
    flt_images = flatten_images(images)
    svm_model = joblib.load('.//model//svm_model.m')
    count = 0
    correct = 0.0
    wrong = 0.0
    for flt in flt_images:
        print('label:' + str(labels[count]))
        pre = svm_model.predict([flt])
        print('prediction:' + str(pre[0]))
        if(pre[0] == labels[count]):
            correct = correct +1.0
        else:
            wrong = wrong + 1.0
        count = count + 1
    return (correct)/(correct + wrong)
    
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC
    model = SVC(kernel='poly',degree=2)
    model.fit(train_x, train_y)
    
    return model 
    
if __name__ == '__main__':
    '''
    train()
    ratio = test()
    print('Accuracy:' + str(ratio))
    '''
    import matplotlib.pyplot as plt

    from skimage.feature import hog
    
    
    image = cv2.imread('no1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    cv2.imshow('origin', image)
    cv2.imshow('hog', hog_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()