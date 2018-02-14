#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:01:24 2018

@author: hendrawahyu
"""

import cv2

# =============================================================================
# template matching
# Params:   gray_temp - template
#           gray_img - image to be matched           
#           methods - 6 methods available (see opencv doc - template matching)
# Output:   center - offset center of matched image
# =============================================================================
def template_matching(gray_img, gray_temp, methods = 'cv2.TM_CCOEFF_NORMED'):
    method = eval(methods)
    height, width = gray_temp.shape         # width and height template
    res = cv2.matchTemplate(gray_img, gray_temp, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if (methods == 'cv2.TM_SQDIFF') or (methods == 'cv2.TM_SQDIFF_NORMED'):
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    center = (int((top_left[0]+bottom_right[0])//2), int((top_left[1]+bottom_right[1])//2))
    return top_left, bottom_right, center
