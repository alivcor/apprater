import cv2, numpy as np
import sys, time, datetime
import FeatureExtractor, EventIssuer
import progressbar
from keras.layers import Flatten, Dense, Input
from keras.models import Sequential
import glob, os, pickle
from keras.layers import Convolution2D, MaxPooling2D
from numpy import genfromtxt
from keras import metrics

alldata = genfromtxt('train.csv', delimiter=',')

trainX = alldata[1:81:,1:10]
trainY = alldata[1:81,10]
testX = alldata[80:,1:10]
testY = alldata[80:,10]

print trainX.shape, trainY.shape
print trainX[0,:]
print trainY[0]

def compileMainModel():
    apprater_model = Sequential()
    apprater_model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    apprater_model.add(Dense(1, kernel_initializer='normal'))
    apprater_model.compile(loss='mean_squared_error', optimizer='adam')
    apprater_model.summary()
    return apprater_model

apprater_model = compileMainModel()
apprater_model.fit(trainX, trainY, batch_size=8, epochs=10)
print apprater_model.predict(trainX[0,:].reshape(1, -1))
print apprater_model.evaluate(x=testX, y=testY)