#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:21:49 2018

@author: hendrawahyu
"""

import numpy as np
from PIL import Image, ImageFilter

# =============================================================================
# The image of my written number has to be formatted in the same way as the images form 
# the MNIST database. If the images don’t match, it will try to predict something else. 
# The MNIST website provides the following information:
# – Images are normalized to fit in a 20×20 pixel box while preserving their aspect ratio.
# – Images are centered in a 28×28 image.
# – Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
# Param: thresh_image - a binary image where background is white and character is black
# Output: return 28 x 28 template with centered characters
# =============================================================================
def prepare_character(bin_img):
    height = bin_img.shape[1]
    width  = bin_img.shape[0]
    newImage = Image.new('L', (28, 28), (255))                          #creates white canvas of 28x28 pixels
    new_img = Image.fromarray(bin_img)
    if width > height:
        nheight = int(round((20.0/width*height), 0))                    #resize
        if(nheight == 0):
            nheight = 1
        elif (nheight < 10):
            new_img = new_img.filter(ImageFilter.SHARPEN)
            offset = ((28 - height)//2, (28 - width) // 2)
        else:
            new_img = new_img.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight)/2),0))
            offset = (4, wtop)
        newImage.paste(new_img, offset)
    else:
        nwidth = int(round((20.0/width*height), 0))                     #resize
        if(nwidth == 0):
            nwidth = 1
        elif (nwidth < 10):
            new_img = new_img.filter(ImageFilter.SHARPEN)
            offset = ((28 - height) // 2, (28 - width) // 2)
        else:
            new_img = new_img.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth)/2),0))
            offset = (wleft, 4)
        newImage.paste(new_img, offset)          
    # Normalize pixel values to 0 and 1 (a MUST for TensorFlow)
    new_img = np.array(newImage)
    return new_img