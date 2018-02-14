#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:58:07 2018

@author: hendrawahyu
"""

import cv2
import numpy as np
from . import normalize_rgb

# =============================================================================
# plant segmentation
# Sources:
#   1.  Su Hnin Hlaing, Aung Soe Khaing, 
#       "Weed and crop segmentation and classification using area thresholding", IJRET eISSN: 2319-1163
#       (http://esatjournals.net/ijret/2014v03/i03/IJRET20140303069.pdf)
#   2.  Various authors,
#       "Vision-Based Row Detection Algorithm Evaluation for Weeding CUltivator Guidance in Lentil"
#       Modern Applied Science, Vol 8, No. 5; 2014, ISSN 1913 - 1844, E-ISSN 1913-1852
# Param:    image - post processed image / region of interest
#           sensitivity - default 0.75, thresholding green value
#           kernel - to remove background noise (open, close)
#           iters - number of iteration for mask (open, close)
#           methods - bilateral (bilateral2) or malvar
# Output:   segmented image
# =============================================================================
def segment_plant(image, sensitivity = 85, kernel_open = (3,3), kernel_close = (3,3), iters = (1,1), methods = 'bilateral'):
    norm = normalize_rgb(image)
    blue = norm[:,:,0]
    green = norm[:,:,1]
    red = norm[:,:,2]
    doubg = np.where((2*green) >= 255, 255, 2*green)
    exg = doubg - red - blue
    if(methods == 'bilateral'):
        processedimg = np.where(exg > sensitivity, exg, 0)
    elif(methods == 'malvar'):
        processedimg = np.where((exg > sensitivity) & (exg < 170), exg, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_open)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_close)
    mask = cv2.morphologyEx(processedimg, cv2.MORPH_OPEN, kernel, iterations = iters[0])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1, iterations = iters[1])
    segmented = cv2.bitwise_and(image, image, mask = mask)
    masked = np.where(segmented == 0, 0, 255).astype('uint8')
    return masked, segmented

# =============================================================================
# grabcut extraction - see open cv documentation
# Param:    img_roi - region of interest
#           dimension - (x, y, w, h) must be equal or less than img_roi
# Output:   segmented image
# =============================================================================
def segment_grabcut(img_roi, dimension, method: cv2.GC_INIT_WITH_RECT):
    mask = np.zeros(img_roi.shape[:2], np.uint8)    #create mask
    bgModel = np.zeros((1,65), np.float64)
    fgModel = np.zeros((1,65), np.float64)
    rect = dimension      # careful selection of value otherwise return black
    cv2.grabCut(img_roi, mask, rect, bgModel, fgModel, 2, method)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    new_roi = img_roi * mask2[:,:,np.newaxis]
    return new_roi