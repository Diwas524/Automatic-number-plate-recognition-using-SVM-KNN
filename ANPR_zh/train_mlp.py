# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:44:03 2016

@author: tallt
"""
import numpy as np
from numpy import *
import cv2
import os
from sklearn.externals import joblib 
IMAGE_WIDTH = 20
IMAGE_LEN = 20

char2dig = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7,
            '8':8, '9':9, 'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15,
            'G':16, 'H':17, 'J':18, 'K':19, 'L':20, 'M':21, 'N':22, 'P':23,
            'Q':24, 'R':25, 'S':26, 'T':27, 'U':28, 'V':29, 'W':30, 'X':31,
            'Y':32, 'Z':33, 'zh_cuan':34, 'zh_e':35, 'zh_gan':36, 'zh_gan1':37,
            'zh_gui':38, 'zh_gui1':39, 'zh_hei':40, 'zh_hu':41, 'zh_ji':42,
            'zh_jin':43, 'zh_jing':44, 'zh_jl':45, 'zh_liao':46, 'zh_lu':47,
            'zh_meng':48, 'zh_min':49, 'zh_ning':50, 'zh_qing':51, 'zh_qiong':52,
            'zh_shan':53, 'zh_su':54, 'zh_sx':55, 'zh_wan':56, 'zh_xiang':57,
            'zh_xin':58, 'zh_yu':59, 'zh_yu1':60, 'zh_yue':61, 'zh_yun':62,
            'zh_zang':63, 'zh_zhe':64}
dig2char = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8',
            9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G',
            17:'H', 18:'J', 19:'K', 20:'L', 21:'M', 22:'N', 23:'P', 24:'Q',
            25:'R', 26:'S', 27:'T', 28:'U', 29:'V', 30:'W', 31:'X', 32:'Y',
            33:'Z', 34:'川', 35:'鄂', 36:'赣', 37:'甘', 38:'贵', 39:'桂',
            40:'黑', 41:'沪', 42:'冀', 43:'津', 44:'京', 45:'吉', 46:'辽',
            47:'鲁', 48:'蒙', 49:'闽', 50:'宁', 51:'青', 52:'琼', 53:'陕',
            54:'苏', 55:'西', 56:'皖', 57:'湘', 58:'新', 59:'豫', 60:'渝',
            61:'粤', 62:'云',63:'藏', 64:'浙'}

def get_train_data(name):
    path = os.getcwd() + '\\resources\\ann\\' + str(name) + '\\'
    images = os.listdir(path)
    image_list = []
    label_list = []
    for image in images:
        img = cv2.imread(path + image)
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #flt_image = reshape(img_gray, IMAGE_WIDTH * IMAGE_LEN)
        image_list.append(img)
        label_list.append(char2dig[name])
    return image_list, label_list
# end fuction

def load_train_data():
    path = os.getcwd() + '\\resources\\ann_train\\'
    file_names = os.listdir(path)
    image_list = []
    label_list = []
    for file_name in file_names:
        image_names = os.listdir(path + file_name)
        for image_name in image_names:
            image = cv2.imread(path + file_name + '\\' + image_name)
            image_list.append(image)
            label_list.append(char2dig[file_name])
    print('data loaded!')
    return image_list, label_list

def load_test_data():
    path = os.getcwd() + '\\resources\\ann_test\\'
    file_names = os.listdir(path)
    image_list = []
    label_list = []
    for file_name in file_names:
        image_names = os.listdir(path + file_name)
        for image_name in image_names:
            image = cv2.imread(path + file_name + '\\' + image_name)
            image_list.append(image)
            label_list.append(char2dig[file_name])
    print('data loaded!')
    return image_list, label_list

def train():
    images, labels = load_train_data()
    flt_images = []
    count = 0
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flt_image = reshape(gray_image, 20 * 20)
        print('size:' + str(flt_image.size) + ' label:' + str(labels[count]))
        count = count + 1
        flt_images.append(flt_image)
    from sklearn.neighbors import KNeighborsClassifier
    KNN_model = KNeighborsClassifier(n_neighbors=3)
    KNN_model.fit(flt_images,labels)
    print('model training completed!')
    joblib.dump(KNN_model, './/model//KNN_model.m')
    return KNN_model

def test():
    images, labels = load_test_data()
    KNN_model = joblib.load('.//model//KNN_model.m')
    count = 0
    correct = 0.0
    wrong = 0.0
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flt = reshape(gray, 20 * 20)
        print('label:' + str(labels[count]))
        pre = KNN_model.predict([flt])
        print('predict:' + str(pre[0]))
        if pre[0] == labels[count]:
            correct = correct + 1.0
        else :
            wrong = wrong + 1.0
        count = count + 1
    ratio = (correct) / (correct + wrong)
    return ratio
        
def KNN_predict(image):
    KNN_model = joblib.load('.//model//KNN_model.m')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resize_image = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_CUBIC)
    flt = reshape(resize_image, 20 * 20)
    return KNN_model.predict([flt])

if __name__ == '__main__':
    # train()
    ratio = test() 
    print('Accuracy:' + str(ratio))