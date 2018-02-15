#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:06:48 2018

@author: hendrawahyu
"""
import cv2
import glob, os
from PlantAnalysis import Plant

# =============================================================================
# Calculate the average of the BGR value
# Param:    index - index of file image    
#           img - 3-channel (BGR) image
#           irimg - 3-channel (BGR) image
# Output:   return class of dict (see save_xlsfile to save into Excel spreadsheet)
# =============================================================================
def pixel_average(index, img, irimg):
    data = {index: dict()}
    data[index] = Plant()
    path = glob.glob('./datafiles/*.dng')
    if (index <= (len(path) - 1)):
        v = os.path.split(path[index])[1]
        v = os.path.splitext(v)[0]
        date,hour = v.split("_")
        blue, green, red, alpha = cv2.mean(img)
        irblue, irgreen, irred, iralpha = cv2.mean(irimg)
        data[index].update({"date": date, "hours": hour, "red": red, "green": green, 
            "blue": blue, "irred": irred, "irgreen": irgreen, "irblue": irblue})
        return data
    else:
        return RuntimeError('file is out of sequence')
