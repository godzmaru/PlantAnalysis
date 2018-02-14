#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:00:44 2017

@author: hendrawahyu
"""

import cv2
import numpy as np
from PlantAnalysis import read_image, template_matching, find_contour, set_ROI
from PlantAnalysis import sharpen_image, predict, prepare_character

search_image = 4        # Issues: 130(charjoined), 121(charjoined), 26(ref upside-down)

# =============================================================================
# CREATE A TEMPLATE FOR ALL
# =============================================================================
image = read_image(0, dir = 'datafiles', ext = 'dng', postprocess = True)[1]

bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)      #find threshold for label
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5), dtype=np.uint8), iterations = 3)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations = 3)

frame = find_contour(opening.copy(), mode  = cv2.RETR_EXTERNAL, find = 'label')
_, _, _,roi_template = set_ROI(bgr, frame, methods = 'rotated', scale = 1.2)
gray = cv2.cvtColor(roi_template, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

# =============================================================================
# IMAGE TO BE MATCHED
# =============================================================================
_, image1 = read_image(search_image, postprocess = True)

bgr2 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR).copy()
bgr2 = cv2.GaussianBlur(bgr2, (5,5), 0)
gray_bgr2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
top_left, bottom_right, center = template_matching(gray_bgr2, gray, methods = 'cv2.TM_SQDIFF_NORMED')
roi_matched = bgr2[top_left[1]+110:bottom_right[1]-110, top_left[0]+20:bottom_right[0]-20]
roi_matched = sharpen_image(roi_matched)

# =============================================================================
# CHARACTER SEGMENTATION
# =============================================================================
gray_roi = cv2.cvtColor(roi_matched, cv2.COLOR_BGR2GRAY)
thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

_, cnts, _ = cv2.findContours(thresh_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
locs = []
characters = {} 
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    #print(w, h, w/float(h))        #debug purpose
    if (w >= 3 and w < 60) and (h > 10 and h < 60):
        cv2.rectangle(roi_matched, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 2)
        locs.append((x, y, w, h))
locs = sorted(locs, key=lambda x:x[0])

read = []
for (i, (cx, cy, cw, ch)) in enumerate(locs):
    roi_char = thresh_roi[cy:cy+ch,cx:cx+cw]
    roi_char = cv2.bitwise_not(roi_char)
    new_roi_char = prepare_character(roi_char)
    characters[i] = new_roi_char
    result = predict(characters[i])
    print(result)
    read_character = result['prediction']
    read.append(read_character)
    
# joined the character into string
read = ''.join(map(str, read))

# =============================================================================
# DISPLAY
# =============================================================================
x_i = roi_matched

cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.resizeWindow("image",500,500)
cv2.imshow("image", x_i)

k = cv2.waitKey(0) & 0xFF
if k == 27:                                         #ESC asciicode
    cv2.destroyAllWindows()     