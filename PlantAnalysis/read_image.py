#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:49:12 2018

@author: hendrawahyu
"""

import rawpy, glob

# =============================================================================
# Read images files using rawpy library
# param:    index    -  index number of image files to open
#           dir      -  directory name (string)
#                       default -> datafiles
#           ext      -  extension of files -> default: dng        
#           demosaic -  boolean False to use BAYER only whereas True is to
#                       process the image up to rawpy.postprocess()
# output:   output_image - pre / post processed image
#           raw_color    - bayer color sequence
# Example:  img = read_image()[2] -> will open default first image on 
#                                    datafiles folder with file ext 'dng'and 
#                                    implement postprocess image    
# =============================================================================
def read_image(index = 0, dir = 'datafiles', ext = 'dng', postprocess = True):
    pathnames = glob.glob('./'+ dir + '/*.' + ext)
    if(index > len(pathnames)-1):
        raise RuntimeError("index out of sequence, load the last index")
    for (i, val) in enumerate(pathnames):
        with rawpy.imread(pathnames[i]) as raw:
            if(postprocess == True):
                image = raw.postprocess()
            else:
                image = raw.raw_image.copy()
            raw_color = raw.color_desc
        if(i == index):
            break
    return raw_color, image

#def get_images(path='./outputs'):
#    images = []
#    for img_path in glob(path + '/*.png'):
#        images.append(cvt_img2np(img_path))
#    return images