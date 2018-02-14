#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:52:58 2018

@author: hendrawahyu
"""

import cv2
import numpy as np


# =============================================================================
# split RGB channels into separate single channel
# param:    image - demosaicking input image
#           single  - True  (dimension: 3)
#                     False (dimension: 1)  
# output:   R, G, B
# example:  green = split_channel(bgr, 'G') -> green channel
# =============================================================================
def split_channel(img, single = True):    
    # set green and red channels to 0 -> blue
    if(single == True):
        b_im = img.copy()
        b_im[:, :, 1] = 0
        b_im[:, :, 2] = 0
        g_im = img.copy()
        g_im[:, :, 0] = 0
        g_im[:, :, 2] = 0
        r_im = img.copy()
        r_im[:, :, 0] = 0
        r_im[:, :, 1] = 0
    else:
        b_im, g_im, r_im = cv2.split(img)
    return b_im, g_im, r_im