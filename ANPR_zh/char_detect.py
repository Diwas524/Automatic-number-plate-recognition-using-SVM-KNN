# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:31:38 2016

@author: tallt
"""
import cv2
import numpy as np
import preprocess as pp 
import os
from sklearn.externals import joblib
from train_mlp import *
PLATE_WIDTH = 400
PLATE_HEIGHT = 150
MIN_AREA = (PLATE_WIDTH/13) * (PLATE_HEIGHT/4) #
MAX_AREA = (PLATE_WIDTH/3) * (PLATE_HEIGHT)
dig2char = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8',
            9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G',
            17:'H', 18:'J', 19:'K', 20:'L', 21:'M', 22:'N', 23:'P', 24:'Q',
            25:'R', 26:'S', 27:'T', 28:'U', 29:'V', 30:'W', 31:'X', 32:'Y',
            33:'Z', 34:'川', 35:'鄂', 36:'赣', 37:'甘', 38:'贵', 39:'桂',
            40:'黑', 41:'沪', 42:'冀', 43:'津', 44:'京', 45:'吉', 46:'辽',
            47:'鲁', 48:'蒙', 49:'闽', 50:'宁', 51:'青', 52:'琼', 53:'陕',
            54:'苏', 55:'西', 56:'皖', 57:'湘', 58:'新', 59:'豫', 60:'渝',
            61:'粤', 62:'云',63:'藏', 64:'浙'}
def threshold_image(image):
    if image is None:
        print('error: image not exit')
        return
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    return threshold
def find_proper_list(rect_list, image):
    if rect_list is None:
        print("error: parameter not exit")
        return
    proper_list = []
    for rect in rect_list:
        (x, y), (width, height), angle = rect

        if angle < -30:
            rect = ((x, y), (height, width), 0)
        if angle > 30:
            rect = ((x, y), (height, width), angle-90)
        (x, y), (width, height), angle = rect
        if width * height < MIN_AREA:
            continue
        if width * height > MAX_AREA:
            continue
        if width > height:
            continue
        
        proper_list.append(rect)
        '''
        image_copy = image
        print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0,255,0),2)
        cv2.imshow('rect', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    return proper_list
# end function
def rect_list_sort(rect_list):
    sorted_rect_list = sorted(rect_list, key = lambda x: x[0][0])
    return sorted_rect_list

def closeure(image):
    if image is None:
        print("error: parameter not exit")
        return
    # end if
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal)
    return closed_image
# end funtion

def load_preprocess_image(image):
    image = cv2.resize(image, (PLATE_WIDTH, PLATE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    image_blurred = pp.gaussian_blur_img(image)
    image_gray = pp.make_gray(image_blurred)
    image_binary = threshold_image(image_gray)
    # image_close = closeure(image_binary)
    contours = pp.find_contour(image_binary)
    rect_list = pp.min_rect(contours)
    proper_list = find_proper_list(rect_list, image)
    return image, contours, proper_list
# end function

def deal_chinest(rect_list):
    sorted_list = rect_list_sort(rect_list)
    (x1, y1), (width1, height1), angle1 = sorted_list[0]
    (x2, y2), (width2, height2), angle2 = sorted_list[1]
    sorted_list[0] = ((x1, y2), (width2, height2), 0)
    return sorted_list

def get_char_in_plate(img):
    image, contours, rect_list = load_preprocess_image(img)
    final_rect_list = deal_chinest(rect_list)
    char_list = []
    for rect in final_rect_list:
        (x, y), (width, height), angle = rect;
        image_ROI = image[y-height/2:y+height/2,x-width/2 - 7:x+width/2 + 7]
        char_list.append(image_ROI)
    return char_list
# end function
def make_gray(image):
    if image is None:
        print("error: parameter not exit")
        return
    # end if
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
# end function
def sobel_process(image):
    if image is None:
        print("error: parameter not exit")
        return
    # end if
    sobel_image = cv2.Sobel(image, -1, 1, 0,ksize=3)
    return sobel_image
# end function

img = cv2.imread('p4.jpg')  
cv2.imshow('',img)
image_list = get_char_in_plate(img)
mlp_model = joblib.load('.//model//KNN_model.m')     
for image in image_list:
    image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_CUBIC)
    gray_image=make_gray(image)
    thres = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 1)
    cv2.imshow('image',thres)
    image_r = np.reshape(thres, 20*20 ) 
    predication = mlp_model.predict([image_r]) 
    print( dig2char[predication[0]] ) 
    cv2.waitKey(0)
cv2.destroyAllWindows()

