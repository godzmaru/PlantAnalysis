#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:00:13 2018

@author: hendrawahyu
"""

import cv2

# =============================================================================
# Normalize image to remove or reduce the effect of light
#   f(x, y) = (R, G, B)
#       total  = (R + G + B)
#       R' = R/total x 255, G' = G/total x 255, B' = B/total x 255
# Param: image -  3 channel image converted as B, G, R (region of interest)
# output: norm_rgb
# =============================================================================
def normalize_rgb(image):
    norm_rgb = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.0
    norm_rgb=cv2.convertScaleAbs(norm_rgb)
    return norm_rgb