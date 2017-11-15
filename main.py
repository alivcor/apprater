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
from keras.constraints import max_norm



def compileMainModel():
    apprater_model = Sequential()
    apprater_model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    apprater_model.add(Dense(9, activation="sigmoid"))
    apprater_model.add(Dense(1, kernel_initializer='normal', kernel_constraint=max_norm(5.)))
    apprater_model.compile(loss='mean_squared_error', optimizer='adam')
    apprater_model.summary()
    return apprater_model

def compileGraphicsModel():
    graphics_model = Sequential()
    graphics_model.add(Dense(9, input_shape=(51,4096), kernel_initializer='normal', activation='relu'))
    graphics_model.add(Flatten())
    graphics_model.add(Dense(1, kernel_initializer='normal', kernel_constraint=max_norm(5.)))
    graphics_model.compile(loss='mean_squared_error', optimizer='adam')
    graphics_model.summary()
    return graphics_model


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
        return obj


def readCSV():
    alldata = genfromtxt('train.csv', delimiter=',')
    trainX = alldata[1:81, 1:10]
    trainY = alldata[1:81, 10]
    testX = alldata[81:, 1:10]
    testY = alldata[81:, 10]
    return trainX, trainY, testX, testY


def loadFeatureVectors():
    feature_vectors = load_obj("feature_vectors")
    feature_vectors_array = []
    for x in feature_vectors:
        if type(x) == int:
            feature_vector = np.zeros((51,4096))
        else:
            feature_vector = np.squeeze(feature_vectors[x])
        # print feature_vector.shape
        feature_vectors_array.append(feature_vector)
    feature_vectors_array = np.stack(feature_vectors_array, axis=0)
    # print feature_vectors_array.shape
    return feature_vectors_array[0:80, :, :], feature_vectors_array[80:, :, :]



trainX, trainY, testX, testY = readCSV()


gtrainX, gtestX = loadFeatureVectors()
print gtrainX.shape, gtestX.shape
graphics_model = compileGraphicsModel()
graphics_model.fit(gtrainX, trainY, batch_size=8, epochs=700)
print graphics_model.evaluate(x=gtestX, y=testY)


#
# apprater_model = compileMainModel()
#
# #
# # print trainX.shape, trainY.shape
# # print trainX[0,:]
# # print trainY[0]
# #
# #
# apprater_model.fit(trainX, trainY, batch_size=8, epochs=100)
#
# print apprater_model.evaluate(x=testX, y=testY)
#
# print apprater_model.predict(trainX[0,:].reshape(1, -1))