#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:56:14 2018

@author: hendrawahyu
"""

import cv2
import numpy as np

# =============================================================================
# find_contour is to join all the continuous points, having the same color/intensity, 
# post thresholding / masking to find region of interest.
# contour retrieval mode and hierarchy (contour approx method)
# Param:    binary_img - binary image
#           mode - see cv2.findContours()
#           method - see cv2.findContours() 
#           find - 'outer', 'label'
# output:   return 2 parameters: area, frame (consist 4 points)
# =============================================================================
def find_contour(binary_img, mode  = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE, find = 'outer'):
    _, contours, hierarchy = cv2.findContours(binary_img.copy(),mode, method)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    frame = None
    # loop over the contours
    if (find == 'outer'):
        minimum, maximum = (2200000, 2500000)
    elif (find == 'label'):
        minimum, maximum = (135000, 150000)
    else:
        print('no contour find under' + find)
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the approximated contour has four points, then assume that the
        # contour is a book -- a book is a rectangle and thus has four vertices
        if len(approx) == 4:
            if minimum < cv2.contourArea(approx) < maximum:
                frame = approx
    return frame
