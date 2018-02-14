#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:12:45 2018

@author: hendrawahyu
"""

import numpy as np
from scipy.misc import imresize
from keras.models import model_from_yaml
import pickle

def load_model(bin_dir):
    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    
    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

# =============================================================================
# This function is predict the characters, the response will be dict with
# prediction and confidence
# Param: characters - binary image contains character with white background
# Output: response of prediction and characters
# =============================================================================
def predict(characters):
    # read parsed image back in 8-bit, black and white mode (L)
    x = np.invert(characters)
    # reshape image data for use in neural network
    x = imresize(x,(28,28))
    x = x.reshape(1,28,28,1)
    # Convert type to float32
    x = x.astype('float32')
    # Normalize to prevent issues with model
    x /= 255
    # Predict from model
    model = load_model('bin')
    mapping = pickle.load(open('%s/mapping.p' % 'bin', 'rb'))
    out = model.predict(x)
    # Generate response
    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
                'confidence': str(max(out[0]) * 100)[:6]}
    return response