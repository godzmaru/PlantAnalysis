#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:56:49 2018

@author: hendrawahyu
"""

import cv2
import numpy as np

# =============================================================================
# Setting inside ROI in 2 methods, fixed distance with 300 as default (for frame 
# or 30pix for label) to constraint frame or calculate using half Euclidean dist.
# Param:    img - post processed image
#           frame - consist of 4 points acquired from contour features
#           dist - default set to 300           
#           methods = none, fixed, euclidean, rotated 
#               None - ONLY to set an outer region of interest (angle invariant)
#               Fixed - set region of interest from the center with fixed distance
#               Euclidean - set region of interest from the center with half Euclidean dist.
#               Rotated - ONLY to set an outer region of interest (angle variant)
# =============================================================================
def set_ROI(img, frame, dist= 300, methods = 'fixed', scale = 1.0, top_view = True):
    angle = 0
    box = None
    # find centroid of frame
    M = cv2.moments(frame)
    c_x = int(M['m10']/M['m00'])        #center position in x coord
    c_y = int(M['m01']/M['m00'])        #center position in y coord
    center = np.array([c_x, c_y])
    if (methods == 'none'):              #finding outside frame
        box = frame
        new_roi = img[np.amin(np.array(frame).T[1])-dist:np.amax(np.array(frame).T[1])+dist, np.amin(np.array(frame).T[0])-dist:np.amax(np.array(frame).T[0])+dist]
        if (top_view == True):
            center, size, warp = rescale_ROI2(new_roi, box)
    elif (methods == 'fixed'):
        new_roi = img[c_y-dist:c_y+dist, c_x-dist: c_x+dist]
        dist = dist
    elif (methods == 'euclidean'):        #finding small frame using half euclidean dist
        x, y, w, h = cv2.boundingRect(frame)
        end = np.array([x, y])
        dist_eu = int((np.sqrt(np.sum((center - end)**2)))/2)
        #getting the roi with euclidean distance and apply no rotation
        new_roi = img[c_y-dist_eu:c_y+dist_eu, c_x-dist_eu: c_x+dist_eu] 
        dist = dist_eu
    elif (methods == 'rotated'):               
        rect = cv2.minAreaRect(frame)          #output: (x,y),(w,h),theta
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        center, size, new_roi = rescale_ROI(img, rect, box, scale = scale)   #scale for display purpose only
        center1 = np.array([c_x, c_y])
        (s_x, s_y) = size
        end = np.array([s_x, s_y])
        dist = int((np.sqrt(np.sum(( center1 - end)**2)))/2)
        angle = rect[2]
    else:
        try:
           methods.lower()
        except UnboundLocalError:
            print('use different method, please put the correct method')
    return (c_x, c_y), dist, angle, new_roi


# =============================================================================
# rescale Region of interest (angle variant - using cv2.minAreaRect()) to
# produce top_bird view
# Param:    image - post processed image
#           rect - gives out 3 params ((x,y)array,(width,height), box angle) 
#           box - 4 points received from contouring using cv2.minAreaRect()
#           scale - scaling the area of ROI (default = 1.0)
# =============================================================================
def rescale_ROI(image, rect, box, scale = 1.0, debug = False):    
    W = rect[1][0]
    H = rect[1][1]
    # get all points and take minimum and maximum point each
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    # set rotated as False and set conditional for top bird view
    rotated = False
    angle = rect[2]
    if angle < -45:
        angle+=90
        rotated = True
    
    center = (int((x1+x2)//2), int((y1+y2)//2))
    size = (int(scale*(x2-x1)),int(scale*(y2-y1)))
    if (debug == True):
        cv2.circle(image, center, 10, (0,255,0), -1)          #for debugging purposes

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv2.getRectSubPix(image, size, center)    
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H 
    croppedH = H if not rotated else W
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*scale), int(croppedH*scale)), (size[0]/2, size[1]/2))
    return center, size, croppedRotated

# =============================================================================
# rescale Region of interest (angle invariant - using cv2.boundingRect() to
# produce top bird view
# Param:    image - post processed image
#           box - 4 points received from contouring
#           scale - scaling the area of ROI (default = 1.0)
# =============================================================================
def rescale_ROI2(image, box):
    points = box.reshape(4,2)
    rect = np.zeros((4,2), dtype = np.float32)
    s = points.sum(axis = 1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis = 1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    center = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
     
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
     
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    # TEMPLATE MATCHING: creating template 
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    height, width, channel = warp.shape
    size = (height, width)
    return center, size, warp