#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:04:09 2018

@author: hendrawahyu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# plot histogram colorspace
# =============================================================================
def plot_histogram_rgb(img, colorspace = 'bgr'):
    plt.subplot(211), plt.hist(img.flatten(),256,[0,256]), plt.xlim([0,256])
    color = tuple(colorspace)
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.subplot(212), plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

# NOT TESTED YET
# =============================================================================
# plot histogram
# =============================================================================
def plot_histogram(img, name=False):
    # get histogram
    if img.dtype == 'uint8':
        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        bins = range(0, 256, 1)

        if name != False:
            # open pyplot plotting window using hist data
            plt.plot(hist)
            # set range of x-axis
            xaxis = plt.xlim([0, 255])
            fig_name = name + '.png'
            # write the figure to current directory
            plt.savefig(fig_name)
            # close pyplot plotting window
            plt.clf()
    else:
        hist, bins = np.histogram(img, bins='auto')
        if name != False:
            # open pyplot plotting window using hist data
            plt.plot(bins[:-1], hist)
            plt.xticks(bins[:-1], rotation='vertical', fontsize=4)
            # set range of x-axis
            # xaxis = plt.xlim([0, bins.max()])
            fig_name = name + '.png'
            # write the figure to current directory
            plt.savefig(fig_name)
            # close pyplot plotting window
            plt.clf()

    return bins, hist