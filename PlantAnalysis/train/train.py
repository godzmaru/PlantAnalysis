#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:20:07 2018

@author: hendrawahyu
"""

import keras
import numpy as np
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.utils import np_utils
from scipy.io import loadmat
import pickle

# =============================================================================
# load matlab data (see how to use below or uncomment)
# =============================================================================
def load_data_mat(mat_file_path, width = 28, height = 28, max_ = None):
    def rotate(img):
        flipped = np.fliplr(img)
        return np.rot90(flipped)
    
    # store content from dataset to mapping.p
    mat = loadmat(mat_file_path)
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('mapping.p', 'wb' ))
    
     # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]
    for i in range(len(training_images)):
        training_images[i] = rotate(training_images[i])

    # Reshape testing data to be valid
    for i in range(len(testing_images)):
        testing_images[i] = rotate(testing_images[i])

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)


# =============================================================================
# Build and train neural network. Also offloads the net in .yaml and the
# weights in .h5 to the bin/.
# =============================================================================
def build_net_simple(training_data, width=28, height=28):
    
    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    # Hyperparameters
    nb_filters = 32 # number of convolutional filters to use
    pool_size = (2, 2) # size of pooling area for max pooling
    kernel_size = (3, 3) # convolution kernel size

    model = Sequential()
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


# =============================================================================
# train data with default epochs = 10
# =============================================================================
def train(model, training_data, callback=True, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if callback == True:
        # Callback for analysis in TensorBoard
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tbCallBack] if callback else None)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open("bin/model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'bin/model.h5')
    print ('TRAINING DONE')


'''
path_file = 'matlab/emnist-byclass.mat'
training_data = load_data_mat(path_file)
model = build_net(training_data)
train(model, training_data, epochs=10)
'''