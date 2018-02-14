#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:54:17 2018

@author: hendrawahyu
"""

import cv2
import numpy as np

# =============================================================================
# equalization Histogram: to stretch pixel count intensities to either ends, thus
# improves the contrast of the image
# Params:   gray_img
#           methods - normal or adaptive (see opencv doc for more info)
# Output:   equ_img
# =============================================================================
def equalize_hist(gray_img, methods = 'normal', clip = 2.0, tile = (8,8)):
    if(methods == 'normal'):
        equ_img = cv2.equalizeHist(gray_img)
    elif(methods == 'adaptive'):
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit= clip, tileGridSize= tile)
        equ_img = clahe.apply(gray_img)
    else:
        try:
           methods.lower()
        except UnboundLocalError:
            print('use different method, please put the correct method')
    return equ_img