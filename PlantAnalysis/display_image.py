#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:03:05 2018

@author: hendrawahyu
"""

import cv2
import numpy as np

# =============================================================================
# displaying image: 
# Param:    image -     displaying the image. Images requires the same dimension
#                       example, [img1, img2, ...] when multi set to True
#           title -     title of the image. default = 'image'
#           multi -     boolean to invoke single or multiple image
#           size  -     size of image (default WxH = 600, 600)
# =============================================================================
def display_image(list_image, title = 'image', size = (600, 600)):
   y = np.hstack(list_image)
   cv2.namedWindow(title,cv2.WINDOW_NORMAL)
   cv2.resizeWindow(title,size)
   cv2.imshow(title, y)
   k = cv2.waitKey(0) & 0xFF
   if k == 27:                                         #ESC asciicode
       cv2.destroyAllWindows()   
