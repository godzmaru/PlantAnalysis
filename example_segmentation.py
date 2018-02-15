#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:21:59 2018

@author: hendrawahyu
"""

import cv2
from PlantAnalysis import read_image, demosaic_bilateral2, find_contour, set_ROI
from PlantAnalysis import template_matching, segment_plant, display_image, plot_histogram_rgb

# list_dir(create = True)         #to create list of files
index_image = 0                           #change this to go to next/prev image
# true color image
image = read_image(index_image, postprocess = False)[1]
bgr = demosaic_bilateral2(image)
bgr_copy = bgr.copy()
gray = cv2.cvtColor(bgr_copy, cv2.COLOR_BGR2GRAY)
 
val, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
# remove any small regions of noise & create a template to match infra red image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 4)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
frame = find_contour(thresh)
cv2.drawContours(bgr_copy, [frame], -1, (255, 0, 0), 30)
center, dist, _, new_roi = set_ROI(bgr_copy, frame, methods = 'euclidean')
cv2.rectangle(bgr_copy, (center[0] - dist, center[1] - dist), (center[0] + dist, center[1] + dist), (0,0,255), 20)
gray_roi = cv2.cvtColor(new_roi, cv2.COLOR_BGR2GRAY)
height, width = gray_roi.shape
# normalise roi with reference area (label)
#avg, new_roi_1 = ref_rescale(new_roi_1, center_1[0], center_1[1], dist = 50)

# =============================================================================
# =============================================================================
# infra red image
image2 = read_image(index_image+1, postprocess = False)[1]
bgr2 = demosaic_bilateral2(image2)
gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)

#cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF and cv2.TM_CCOEFF_NORMED.
top_left, bottom_right, center_2 = template_matching(gray2, gray_roi, methods = 'cv2.TM_CCOEFF_NORMED')
new_roi2 = bgr2[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]] 

#avg, new_roi2 = ref_rescale(new_roi2, center_2[0], center_2[1], dist = 50)
cv2.rectangle(bgr2,top_left, bottom_right, 255, 20)

# =============================================================================
# =============================================================================
#segmentation
mask, segmented = segment_plant(new_roi, sensitivity = 85, kernel_open = (9,9), kernel_close = (3,3), methods = 'bilateral')      #colored image
bgr_copy[center[1]-dist:center[1]+dist, center[0]-dist: center[0]+dist] = segmented
 
res = cv2.bitwise_and(new_roi2, new_roi2, mask = mask[:,:,0])
bgr2[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]] = res

#calculate average and print out result
data = pixel_average(index_image, segmented, res)
save_xlsfile(data, output_file = 'test.xlsx', sheet = 'Sheet1')

display_image([bgr_copy, bgr2], size = (1000, 1000))
plot_histogram_rgb(bgr_copy, colorspace = 'bgr')
plot_histogram_rgb(bgr2, colorspace = 'bgr')
