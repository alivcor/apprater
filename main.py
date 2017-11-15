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
from keras import optimizers, regularizers
from keras.constraints import min_max_norm
import random


def compileMainModel():
    apprater_model = Sequential()
    apprater_model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
    apprater_model.add(Dense(10, activation="sigmoid"))
    # apprater_model.add(Dense(1, kernel_initializer='normal', kernel_constraint=min_max_norm(min_value=0.0, max_value=5.0)))
    apprater_model.compile(loss='mean_squared_error', optimizer='adam')
    apprater_model.summary()
    return apprater_model

def compileGraphicsModel():
    graphics_model = Sequential()
    graphics_model.add(Dense(9, input_shape=(51,4096), kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    graphics_model.add(Flatten())
    graphics_model.add(Dense(1, kernel_initializer='normal', kernel_constraint=min_max_norm(min_value=0.0, max_value=5.0), kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
    graphics_model.compile(loss='mean_squared_error', optimizer='adam')
    graphics_model.summary()
    return graphics_model


def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    return True


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
        return obj


def randPartition(alldata_X, alldata_gX, alldata_Y, _FRACTION):
    """
    alldata_X : All of your X (Features) data
    alldata_Y : All of your Y (Prediction) data
    _FRACTION : The fraction of data rows you want for train (0.75 means you need 75% of your data as train and 25% as test)
    """
    np.random.seed(0)
    indices = np.arange(alldata_X.shape[0]-1)
    np.random.shuffle(indices)

    dataX = alldata_X[indices]
    gdataX = alldata_gX[indices]
    dataY = alldata_Y[indices]

    partition_index = int(dataX.shape[0] * _FRACTION)

    trainX = dataX[0:partition_index]
    gtrainX = gdataX[0:partition_index]
    # testX = dataX[partition_index:dataX.shape[0]]
    # gtestX = gdataX[partition_index:gdataX.shape[0]]
    testX = dataX[partition_index:partition_index+150]
    gtestX = gdataX[partition_index:partition_index+150]

    trainY = dataY[0:partition_index]
    # testY = dataY[partition_index:dataY.shape[0]]
    testY = dataY[partition_index:partition_index+150]

    return [trainX, trainY, testX, testY, gtrainX, gtestX]



def readCSV():
    alldata = genfromtxt('train.csv', delimiter=',')
    alldataX = alldata[:,1:10]
    alldataY = alldata[:, 10]
    # trainX = alldata[1:81, 1:10]
    # trainY = alldata[1:81, 10]
    # testX = alldata[81:, 1:10]
    # testY = alldata[81:, 10]
    return alldataX, alldataY


def loadFeatureVectors():
    feature_vectors = load_obj("feature_vectors_complete")
    feature_vectors_array = []
    for x in feature_vectors:
        if type(x) == int:
            feature_vector = np.ones((51,4096))
        else:
            feature_vector = np.squeeze(feature_vectors[x])
        # print feature_vector.shape
        feature_vectors_array.append(feature_vector)
    feature_vectors_array = np.stack(feature_vectors_array, axis=0)
    # print feature_vectors_array.shape
    return feature_vectors_array


def loadDataset():
    alldataX, alldataY = readCSV()
    gdataX = loadFeatureVectors()
    trainX, trainY, testX, testY, gtrainX, gtestX = randPartition(alldataX, gdataX, alldataY, 0.50)
    print trainX.shape, trainY.shape, testX.shape, testY.shape, gtrainX.shape, gtestX.shape
    return trainX, trainY, testX, testY, gtrainX, gtestX


trainX, trainY, testX, testY, gtrainX, gtestX = loadDataset()

# graphics_model = compileGraphicsModel()
# graphics_model.fit(gtrainX, trainY, batch_size=12, epochs=200)
# print graphics_model.evaluate(x=gtestX, y=testY)
#
# graphic_model_train_outputs = graphics_model.predict(gtrainX)
#
# save_obj(graphic_model_train_outputs, "graphic_model_train_outputs")

graphic_model_train_outputs = load_obj("graphic_model_train_outputs")
print graphic_model_train_outputs.shape
#
trainX = np.hstack((trainX, graphic_model_train_outputs))

print trainX.shape
# graphic_model_test_outputs = graphics_model.predict(gtestX)
#
# testX = np.hstack((trainX, graphic_model_test_outputs))

# #
# # print trainX.shape, trainY.shape
# # print trainX[0,:]
# # print trainY[0]
# #
# #
print trainX
# apprater_model = compileMainModel()
# apprater_model.fit(trainX, trainY, batch_size=12, epochs=100)
# #
# print apprater_model.evaluate(x=testX, y=testY)
#
# print apprater_model.predict(trainX[0,:].reshape(1, -1))