#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:44:25 2018

@author: hendrawahyu
"""

import cv2
import numpy as np
from scipy.ndimage.filters import convolve, correlate

# =============================================================================
# demosaicking / debayering image using simple bilinear interpolation
# param:    image - bayer input image ONLY (see read_image 
#                   postprocess MUST be FALSE)
# output:   rgb image  
# =============================================================================
def demosaic_bilateral(img):
    col_image = np.zeros(img.shape + (3,), dtype = np.uint16)     # create 3 channels
    col_image[::2,  ::2,1] = img[::2,::2]                         # red pixels
    col_image[::2, 1::2,0] = img[::2,1::2]                        # green pixels
    col_image[1::2, ::2,1] = img[1::2,1::2]                       # green pixels
    col_image[1::2,1::2,2] = img[1::2,::2]                        # blue pixels

    # convolution kernels
    # green
    H_G = np.asarray([[0, 1, 0],[1, 4, 1],[0, 1, 0]], dtype = np.float16)/4
    # red/blue
    H_RB = np.asarray([[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype = np.float16)/4

    # bilateral interpolation    
    G = correlate(col_image[:,:,1], H_G) 
    R = correlate(col_image[:,:,0], H_RB)
    B = correlate(col_image[:,:,2], H_RB)
    
    # fix orientation after convolution
    r = R[::-1].T       # transpose red
    b = B[::-1].T       # transpose blue
    g = G[::-1].T       # transpose green
    
    #combine all colors together in BGR format (openCV)
    bgr = np.dstack((b,g,r))
    bgr = cv2.convertScaleAbs(bgr)       # convert to uint8
    bgr = cv2.GaussianBlur(bgr, (5,5), 0)
    return bgr


# =============================================================================
# demosaicking / debayering image using simple bilinear interpolation different 
# convolution kernels
# Param:    image - bayer input image ONLY (see read_image 
#                   postprocess MUST be FALSE)
# Output:   rgb image  
# =============================================================================
def demosaic_bilateral2(image):
    col_image = np.zeros(image.shape + (3,), dtype = np.uint16)     # create 3 channels
    col_image[::2,  ::2,1] = image[::2,::2]                         # red pixels
    col_image[::2, 1::2,0] = image[::2,1::2]                        # green pixels
    col_image[1::2, ::2,1] = image[1::2,1::2]                       # green pixels
    col_image[1::2,1::2,2] = image[1::2,::2]                        # blue pixels 
    
    H_G = np.asarray([[0, 1, 0],[1, 0, 1],[0, 1, 0]], dtype = np.float16)/2
    H_RB = np.asarray([[1, 0, 1],[0, 0, 0],[1, 0, 1]],dtype = np.float16)/2
    
    #bilateral interpolation
    #green    
    G1 = (col_image[:,:,1] + convolve(col_image[:,:,1], H_G))
    #blue
    B1 = convolve(col_image[:,:,2], H_RB)
    B2= convolve(col_image[:,:,2] + B1, H_G)
    Bt= (col_image[:,:,2] + B1 + B2)
    #red
    R1 = convolve(col_image[:,:,0], H_RB)
    R2= convolve(col_image[:,:,0] + R1, H_G)
    Rt = (col_image[:,:,0] + R1 + R2)

    r = Rt[::-1].T
    b = Bt[::-1].T
    g = G1[::-1].T

    bgr = np.dstack((b,g,r))
    bgr = cv2.convertScaleAbs(bgr)
    bgr = cv2.GaussianBlur(bgr, (5,5), 0)
    return bgr

# =============================================================================
# demosaicking / debayering image using malvar, cutler, he interpolation
# =============================================================================
def demosaic_malvar(img):
    col_image = np.zeros(img.shape + (3,), dtype = np.uint16)     # create 3 channels
    col_image[::2,  ::2,1] = img[::2,::2]                        # red pixels
    col_image[::2, 1::2,0] = img[::2,1::2]                       # green pixels
    col_image[1::2, ::2,2] = img[1::2,1::2]                      # green pixels
    col_image[1::2,1::2,1] = img[1::2,::2]                       # blue pixels

    R = col_image[:,:,0]
    G = col_image[:,:,1]
    B = col_image[:,:,2]

    # convolution kernels
    # Green at red location and green at blue location
    GR_GB = np.asarray([[0, 0, -1, 0, 0],[0, 0, 2, 0, 0],[-1, 2, 4, 2, -1],[0, 0, 2, 0, 0],[0, 0, -1, 0, 0]], dtype = np.float16)/4
    # Red at green location in red rows and Blue at green locations in red rows
    Rg_RB_Bg_BR = np.asarray([[0, 0, 0.5, 0, 0],[0, -1, 0, -1, 0],[-1, 4, 5, 4, -1],[0, -1, 0, -1, 0],[0, 0, 0.5, 0, 0]], dtype = np.float16)/4
    # R at green locations in blue rows and Blue at green locations in blue rows
    Rg_BR_Bg_RB = np.asarray([[0, 0, -1, 0, 0],[0, -1, 4, -1, 0], [0.5, 0, 5, 0, 0.5], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0]], dtype = np.float16)/4
    # Red at blue location and Blue at red locations
    Rb_BB_Br_RR = np.asarray([[0, 0, -1.5, 0, 0],[0, 2, 0, 2, 0],[-1.5, 0, 6, 0, -1.5],[0, 2, 0, 2, 0],[0, 0, -1.5, 0, 0]], dtype = np.float16)/4

    G = np.where(np.logical_or(R != 0, B != 0), convolve(img, GR_GB), G)
    RBg_RBBR = convolve(img, Rg_RB_Bg_BR)
    RBg_BRRB = convolve(img, Rg_BR_Bg_RB)
    RBgr_BBRR = convolve(img, Rb_BB_Br_RR)

    # Red rows.
    R_r = np.transpose(np.any(R != 0, axis=1)[np.newaxis]) * np.ones(R.shape).astype(np.uint8)
    # Red / Blue columns.
    R_c = np.any(R != 0, axis=0)[np.newaxis] * np.ones(R.shape).astype(np.uint8)
    # Blue rows.
    B_r = np.transpose(np.any(B != 0, axis=1)[np.newaxis]) * np.ones(B.shape).astype(np.uint8)
    # Blue columns
    B_c = np.any(B != 0, axis=0)[np.newaxis] * np.ones(B.shape).astype(np.uint8)

    R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    r = R[::-1].T
    b = B[::-1].T 
    g = G[::-1].T 

    bgr = np.dstack((b,g,r))
    bgr = cv2.convertScaleAbs(bgr)                                 #convert to uint8
    bgr = cv2.GaussianBlur(bgr, (5,5), 0)
    return bgr