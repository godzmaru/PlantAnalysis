#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:17:42 2018

@author: hendrawahyu
"""

import os, glob, csv
import argparse

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
def list_dir(dir_name = 'datafiles', ext = 'dng', create = False):
    pathnames = glob.glob('./'+ dir_name + '/*.' + ext)
    if(create == False):
        for (i, val) in enumerate(pathnames):
            print('{0:2d} {1:3s}'.format(i, val))
    else:
        with open('list.csv', 'w') as csvfile:
            fieldnames = ['index', 'date', 'hours']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for (i, val) in enumerate(pathnames):
                v = os.path.split(val)[1]
                v = os.path.splitext(v)[0]
                date,hours = v.split("_")
                writer.writerow({'index': i, 'date': date, 'hours':hours}) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='datafiles', type=str, help='Path to image folder')
    parser.add_argument('--ext', default='dng', type=str, help='Image Extension')
    parser.add_argument('--save', default = False,  type=bool, help='create csv of file list')
    args = parser.parse_args()

    list_dir(args.file)
    