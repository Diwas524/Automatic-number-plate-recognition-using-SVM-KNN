# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:34:12 2016

@author: tallt
"""
import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main():
    imgTrainingNumbers = cv2.imread("training_chars.png")
    
    
    if imgTrainingNumbers is None:
        print ("error: img not exit\n")
        os.system("pause")
        return
    # end if
    
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # get gray img
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)

    # filter img to binary form
    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)
    #cv2.imshow("imgThresh", imgThresh)
    imgThreshCopy = imgThresh.copy()
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.imshow('contours', imgContours)

    # declare empty numpy array, it will store img data
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    
    # declare empty array to store input
    intClassifications = []
    intValidChars = [ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),
                     ord('7'),ord('8'),ord('9'),ord('A'),ord('B'),ord('C'),ord('D'),
                     ord('E'),ord('F'),ord('G'),ord('H'),ord('I'),ord('J'),ord('K'),
                     ord('L'),ord('M'),ord('N'),ord('O'),ord('P'),ord('Q'),ord('R'),
                     ord('S'),ord('T'),ord('U'),ord('V'),ord('W'),ord('X'),ord('Y'),
                     ord('Z'),ord('京'),ord('津'),ord('沪'),ord('渝'),ord('蒙'),ord('新'),
                     ord('藏'),ord('宁'),ord('桂'),ord('黑'),ord('吉'),ord('辽'),ord('晋'),
                     ord('冀'),ord('港'),ord('澳'),ord('青'),ord('鲁'),ord('豫'),ord('苏'),
                     ord('皖'),ord('浙'),ord('闽'),ord('赣'),ord('湘'),ord('鄂'),ord('粤'),
                     ord('琼'),ord('甘'),ord('陕'),ord('黔'),ord('滇'),ord('川')]
 
    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect

                                                # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,           # draw rectangle on original training image
                          (intX, intY),                 # upper left corner
                          (intX+intW,intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage

            #cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it

            intChar = cv2.waitKey(0)                     # get key press
#           intChar = (input()
            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .

                intClassifications.append(intChar)                                                # append classification char to integer list of chars (we will convert to float later before writing to file)

                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
            # end if
        # end if
    # end for
    
    
    fltClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))
    
    print ("\ntraining complete~\n")
    
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)
    
    cv2.destroyAllWindows()
    
    return
    

if __name__ == "__main__":
    main()
#end if


