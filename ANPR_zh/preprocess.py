# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:18:42 2016

@author: tallt
"""
PLATE_HEIGHT = 136
PLATE_WIDTH = 36
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.svm import SVC
def load_image(image_name):
    img = cv2.imread(image_name)
    img = cv2.resize(img, (800, 480), interpolation=cv2.INTER_CUBIC)
    if img is None:
        print("error: file not exit!!")
        return
    # end if
    return img
# end function
def color_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    lower_blue = np.array([100,60,60])
    upper_blue = np.array([140,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(image,image, mask= mask) 
    return res
def gaussian_blur_img(image):
    if image is None:
        print("error: parameter not exit")
        return
    # end if
    blur = cv2.GaussianBlur(image, (3,3), 0)
    return blur
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

def make_binary(image):
    if image is None:
        print("error: parameter not exit")
        return
    # end if
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 41, 15)
    return threshold
# end function

def closure(image):
    if image is None:
        print("error: parameter not exit")
        return
    # end if
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal)
    return closed_image
# end function

def find_contour(image):
    if image is None:
        print("error: parameter not exit")
        return
    # end if
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255,255,255), 1)
    return contours
# end function

def verify_plate(rect_list,img):
    if rect_list is None:
        print("error: parameter not exit")
        return
    # end if
    svm_model = joblib.load('.//model//svm_model.m') 
    proper_list = []  
    count = 0
    for rect in rect_list:
        (x, y), (width, height), angle = rect
        if angle > 30 or angle < -30:
            continue
        elif width < height:
            continue
        elif height == 0:
            continue
        elif width < 30 or height < 20:
            continue
        elif width / height < 2:
            continue
        #
        count=0
        image = img[y-height/2:y+height/2,x-width/2:x+width/2] 
        image = cv2.resize(image, (136, 36), interpolation=cv2.INTER_CUBIC)
        img_c = color_image(image) 
        count=0
        for i in range(36):
            for j in range(136):
                b,g,r = img_c[i][j]
                if(  b!=0 or r!=0 or g!=0):
                        count+=1
        percent=float(count)/(36*136) 
        if(percent < 0.1 ):
            continue
         
       
                
         
        
#        image_r = reshape(image, 136 * 36 *3) 
#        predication = svm_model.predict(image_r) 
#        if predication[0] == 0: 
#            continue
        rect = ((x, y), (width, height),0)
        proper_list.append(rect)
        print(rect) 
         
    return proper_list
# end function

def min_rect(contours):
    if contours is None:
        print("error: para empty")
        return
    rect_list = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        rect_list.append(rect)
    return rect_list
# end function

def draw_rect(image, rect_list):
    for rect in rect_list:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0,255,0),2)
    return
# end function

def resize_rect(image):
    if image is None:
        print('error: para empty')
        return
    resized_image = cv2.resize(image, (PLATE_HEIGHT, PLATE_WIDTH), interpolation=cv2.INTER_CUBIC)
    return resized_image
# end function

def get_ROI_image(image, rect_list):
    image_list = []
    for rect in rect_list:
        (x, y), (width, height), angle = rect;
        image_ROI = image[y-height/2:y+height/2,x-width/2:x+width/2]
        image_list.append(image_ROI)
    return image_list
# end function

def roi_resize(roi_list):
    proper_roi_list = []
    for roi_image in roi_list:
        roi_image = resize_rect(roi_image)
        proper_roi_list.append(roi_image)
    return proper_roi_list
# end function

def extract_image(imageName):
    img = load_image(imageName)  #读取
    cv2.imshow('img', img)
    cv2.waitKey()
    img_copy = img
    img_blurred = gaussian_blur_img(img)  #高斯
    img_gray = make_gray(img_blurred)     #转换为灰阶图
    cv2.imshow('img_gray', img_gray)
    cv2.waitKey()
    img_sobel = sobel_process(img_gray)
    cv2.imshow('img_sobel', img_sobel)    #sobel算子
    cv2.waitKey()
    image_threshold = make_binary(img_sobel)
    cv2.imshow('image_threshold', image_threshold)  #转化2值图像
    cv2.waitKey()
    image_closed = closure(image_threshold)
    image_closed_copy = image_closed
    cv2.imshow('image_closed', image_closed)
    cv2.waitKey()
    contours = find_contour(image_closed_copy)
    #cv2.drawContours(img, contours, -1, (0,0,255),2)
    rect_list = min_rect(contours)   #找出所有轮廓
    proper_rect_list = verify_plate(rect_list,img_copy)  #判断
    draw_rect(img, proper_rect_list)
    cv2.imshow('img', img)
    cv2.waitKey()
    roi_list = get_ROI_image(img_copy, proper_rect_list)  #截取
    proper_roi_list = roi_resize(roi_list)
    
    
    return proper_roi_list
# end function
def extract_image_color(imageName):
    img = load_image(imageName) 
    img_copy=img
    cv2.imshow('original',img)
    cv2.waitKey()
    img_c = color_image(img) 
    cv2.imshow('blue',img_c)
    cv2.waitKey()
    
    for i in range(480):
        for j in range(800):
            b,g,r = img_c[i][j]
            if(  b!=0 or r!=0 or g!=0):
                    img_c[i][j]=(255,255,255)
    
    img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',img_gray)
    cv2.waitKey()
    _ , img_threshold=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    cv2.imshow('threshold',img_threshold)
    cv2.waitKey()
    img_closed = closure(img_threshold)
    cv2.imshow('closure',img_closed)
    cv2.waitKey()
    img_closed_copy = img_closed
    contours = find_contour(img_closed_copy)
    rect_list = min_rect(contours)   #找出所有轮廓
    proper_rect_list = verify_plate(rect_list,img_copy)  #判断
    draw_rect(img, proper_rect_list) 
    roi_list = get_ROI_image(img_copy, proper_rect_list)  #截取
    proper_roi_list = roi_resize(roi_list)
    return proper_roi_list
    
    
if __name__ == '__main__': 
    
    image_name = '.\\resources\\image\\plate2.jpg'
    image_list = extract_image_color(image_name)
    KNN_model = joblib.load('.//model//KNN_model.m') 
    find_flag=0
    for image in image_list:
        cv2.imshow('roi', image)
        find_flag=1
        cv2.waitKey(0) 
    if(find_flag == 0): 
        cv2.destroyAllWindows()
        image_list = extract_image(image_name)
        for image in image_list:
            cv2.imshow('roi', image) 
            cv2.waitKey(0) 
            
    for image in image_list:
        cv2.imshow('plate!!',image)
        print('plate!!!!!!')
        cv2.imwrite('.//plate1.jpg', image)
        cv2.waitKey(0) 
        
    cv2.destroyAllWindows()
    cv2.waitKey(0) 
    
