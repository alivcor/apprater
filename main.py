import cv2, numpy as np
import sys, time, datetime
import FeatureExtractor, EventIssuer
import progressbar
from keras.layers import Flatten, Dense, Input
from keras.models import Sequential
import glob, os, pickle
from keras.layers import Convolution2D, MaxPooling2D



def compileMainModel():
    apprater_model = Sequential()
    apprater_model.add(Dense(12, input_dim=11, activation='relu', name='rating'))
    apprater_model.add(Dense(8, activation='tanh'))
    apprater_model.add(Dense(8, activation='relu'))
    apprater_model.add(Dense(5, activation='softmax'))
    apprater_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    apprater_model.summary()

compileMainModel()