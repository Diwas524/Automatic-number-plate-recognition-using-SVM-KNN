# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:27:12 2016

@author: tallt
"""

import cv2
import numpy as np

class possiblePlate:
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None 
        
        self.rrLocationOfPlateInScene = None 
        
        self.strChars = ""