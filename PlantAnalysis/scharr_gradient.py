#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:14:18 2018

@author: hendrawahyu
"""

import cv2
import numpy as np

# =============================================================================
# scharr_gradient is to calculae the first x and y image derivative using
# Scharr operator. The image requires a sobel operator first as a discrete
# differentiation operator to compute an approx of the gradient of an image
# intensity function.
# Param:    bin_img     - binarized image
#           dx          - set this to 1 (dy = 0) to get gradient X (default)
#           dy          - set this to 1 (dx = 0) to get gradient Y
# Output:   return gradientX or gradientY
# =============================================================================
def scharr_gradient(bin_img, dx = 1, dy = 0):
    grad = cv2.Sobel(bin_img, ddepth=cv2.CV_32F, dx=dx, dy=dy,ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (255 * ((grad - minVal) / (maxVal - minVal)))
    grad = grad.astype("uint8")
    return grad