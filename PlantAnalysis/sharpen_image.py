#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:06:40 2018

@author: hendrawahyu
"""
import cv2
import numpy as np

# =============================================================================
# This function is used to sharpen the image using a simple kernel
# =============================================================================
def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    return image