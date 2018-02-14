#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:02:06 2018

@author: hendrawahyu
"""

import cv2
import numpy as np
from . import display_image

# =============================================================================
# rescale all color channels in ROI with reference label / marker
# Param:    image - 3 channels colored image
#           c_x - center of x coordinate of ROI
#           c_y - center of y coordinate of ROI
#           dist - size of sample M x N reference label / marker (default = 50)
#           debug - plot a circle point and rectangular to indicate label/marker area
# Output:   rescale_img = image / average(reference) * 255
# =============================================================================
def ref_rescale(image, c_x, c_y, dist = 50, debug = False):
    rescale_img = np.zeros_like(image, dtype = np.float32)
    cx = c_x + 810
    cy = c_y
    if(debug == True):
        cv2.circle(image,  (cx, cy), 15, (0,0,255), -1)
        cv2.rectangle(image, (cx - dist, cy - dist), (cx + dist, cy + dist), (0,0,255), 10)
        display_image([image], title = 'image', size = (600, 600))
    label_roi = image[cy-dist:cy+dist, cx-dist: cx+dist]
    average = cv2.mean(label_roi)       #gives blue, green, red average
    rescale_img = (image / average) * 255.0
    rescale_img = cv2.convertScaleAbs(rescale_img)
    return average, rescale_img