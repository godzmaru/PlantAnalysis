#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 01:06:47 2018

@author: hendrawahyu
"""
import pandas as pd

# =============================================================================
# Save class Plant data into excel File
# Param:    data - dicts of dict input
#           output_file - name of output file, default = 'test.xlsx'
#           sheet - excel worksheet, default = 'Sheet1'
# =============================================================================
def save_xlsfile(data, output_file = 'test.xlsx', sheet = 'Sheet1'):
    df = pd.DataFrame.from_dict(data, orient = 'index')
    writer = pd.ExcelWriter(output_file)
    df.to_excel(writer, sheet)  
    writer.save()
    return