import cv2, numpy as np
import sys, time, datetime
import FeatureExtractor, EventIssuer
import progressbar
from keras.layers import Flatten, Dense, Input
from keras.models import Sequential
from keras.models import load_model
import glob, os, pickle
from keras.layers import Convolution2D, MaxPooling2D
from numpy import genfromtxt
import matplotlib.pyplot as plt
from keras import metrics
from keras import optimizers, regularizers
from keras.constraints import min_max_norm
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import random


def compileMainModel():
    apprater_model = Sequential()
    apprater_model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
    apprater_model.add(Dense(8, activation="relu"))
    apprater_model.add(Dense(1, kernel_initializer='normal', kernel_constraint=min_max_norm(min_value=0.0, max_value=5.0), kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    apprater_model.compile(loss='mean_squared_error', optimizer=adam)
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
    # np.random.shuffle(indices)

    dataX = alldata_X[indices]
    gdataX = alldata_gX[indices]
    dataY = alldata_Y[indices]

    partition_index = int(dataX.shape[0] * _FRACTION)

    trainX = dataX[0:partition_index]
    gtrainX = gdataX[0:partition_index]
    testX = dataX[partition_index:dataX.shape[0]]
    gtestX = gdataX[partition_index:gdataX.shape[0]]
    # testX = dataX[partition_index:partition_index+150]
    # gtestX = gdataX[partition_index:partition_index+150]

    trainY = dataY[0:partition_index]
    testY = dataY[partition_index:dataY.shape[0]]
    # testY = dataY[partition_index:partition_index+150]

    return [trainX, trainY, testX, testY, gtrainX, gtestX]



def readCSV():
    alldata = genfromtxt('train_v2.csv', delimiter=',')
    alldataX = alldata[1:,1:10]
    alldataY = alldata[1:, 10]
    # trainX = alldata[1:81, 1:10]
    # trainY = alldata[1:81, 10]
    # testX = alldata[81:, 1:10]
    # testY = alldata[81:, 10]
    return alldataX, alldataY


def loadFeatureVectors():
    feature_vectors = load_obj("feature_vectors_complete")
    feature_vectors_array = []
    fint = 0
    for x in feature_vectors:
        if type(feature_vectors[x]) == int or feature_vectors[x].shape != (51, 1, 4096):
            fint += 1
            # try:
            #     print np.amax(feature_vectors[x].flatten()), np.amin(feature_vectors[x].flatten())
            # except:
            #     pass
            feature_vector = np.ones((1, 51, 4096))
        else:
            feature_vector = np.array([np.squeeze(feature_vectors[x])])
        # print feature_vector.shape
        feature_vectors_array.append(feature_vector)
    feature_vectors_array = np.squeeze(np.array(feature_vectors_array))
    # print "feature_vectors_array.shape", feature_vectors_array.shape
    # print "found", fint, "fints"
    return feature_vectors_array


def loadDataset():
    alldataX, alldataY = readCSV()
    gdataX = loadFeatureVectors()
    trainX, trainY, testX, testY, gtrainX, gtestX = randPartition(alldataX, gdataX, alldataY, 0.80)
    print trainX.shape, trainY.shape, testX.shape, testY.shape, gtrainX.shape, gtestX.shape
    return trainX, trainY, testX, testY, gtrainX, gtestX


def plot_hist(x):
    n, bins, patches = plt.hist(x)
    plt.show()
    # sys.exit(0)
    pass


trainX, trainY, testX, testY, gtrainX, gtestX = loadDataset()

print np.amax(gtrainX[10,:,:].flatten()), np.amin(gtrainX[10,:,:].flatten())


# # MARK: GRAPHICS MODEL TRAINING
# graphics_model = compileGraphicsModel()
# graphics_model.fit(gtrainX, trainY, batch_size=12, epochs=120)
# graphics_model.save("obj/trained_graphic_model.h5")


# # MARK: LOAD GRAPHICS MODEL
graphics_model = load_model("obj/trained_graphic_model.h5")
print "graphics_model.evaluate(x=gtestX, y=testY)", graphics_model.evaluate(x=gtestX, y=testY)
graphic_model_train_outputs = graphics_model.predict(gtrainX)
save_obj(graphic_model_train_outputs, "graphic_model_train_outputs")
print "Evaulation: "
print graphics_model.evaluate(x=gtestX, y=testY)


# # MARK: GRAPHICS MODEL OUTPUT LOADING
graphic_model_train_outputs = load_obj("graphic_model_train_outputs") #t
print "graphic_model_train_outputs.shape", graphic_model_train_outputs.shape #t


# # MARK: APPEND GRAPHICS MODEL OUTPUT WITH TRAIN.CSV INPUT
trainX = np.hstack((trainX, graphic_model_train_outputs)) #t
print "trainX.shape", trainX.shape  #t


# # MARK: DO SAME FOR TEST
graphic_model_test_outputs = graphics_model.predict(gtestX)
testX = np.hstack((testX, graphic_model_test_outputs))
print "testX.shape", testX.shape


# # MARK: TRAIN MAIN MODEL
# apprater_model = compileMainModel()
# apprater_model.fit(trainX, trainY, batch_size=32, epochs=1000)
# apprater_model.save("obj/apprater_model.h5")
# print "Evaulation: "
# print apprater_model.evaluate(x=testX, y=testY)
# print "\n"
# print "Predicted Output: ", apprater_model.predict(trainX[0,:].reshape(1, -1))
#


print "\n\nLinear Regression:\n"
# Create linear regression object
linear_regr = linear_model.LinearRegression()
# Train the model using the training sets
linear_regr.fit(trainX, trainY)
# Make predictions using the testing set
pred_y = linear_regr.predict(testX)
# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.6f"
      % mean_squared_error(testY, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f' % r2_score(testY, pred_y))




# linear_regr_stdzd = linear_model.LinearRegression()
# linear_regr_stdzd.fit(trainX / np.std(trainX, 0), trainY)
# influence_val = linear_regr_stdzd.coef_



print "\n\nRidge Regression: \n"
ridge_regr = linear_model.Ridge(alpha =.7)
# Train the model using the training sets
ridge_regr.fit(trainX, trainY)
# Make predictions using the testing set
pred_y = ridge_regr.predict(testX)
# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.6f"
      % mean_squared_error(testY, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f' % r2_score(testY, pred_y))





print "\n\nLasso Regression: \n"
lasso_regr = linear_model.Lasso(alpha =.1,  max_iter=10000)
# Train the model using the training sets
lasso_regr.fit(trainX, trainY)
# Make predictions using the testing set
pred_y = lasso_regr.predict(testX)
# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.6f"
      % mean_squared_error(testY, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f' % r2_score(testY, pred_y))






print "\n\nRandom Forest Regression: \n"
rf_regr = RandomForestRegressor(max_depth=2000, random_state=0)
rf_regr.fit(trainX, trainY)
# print(regr.feature_importances_)
# Make predictions using the testing set
pred_y = rf_regr.predict(testX)
# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.6f"
      % mean_squared_error(testY, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f' % r2_score(testY, pred_y))





print "\n\nK Nearest Neighbour Regression: \n"
neigh = KNeighborsRegressor(8)
neigh.fit(trainX, trainY)
# Make predictions using the testing set
pred_y = neigh.predict(testX)
# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.6f"
      % mean_squared_error(testY, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f' % r2_score(testY, pred_y))




print "\n\nElastic Net Regression: \n"
elastic_net_regr = ElasticNet(random_state=0)
elastic_net_regr.fit(trainX, trainY)
# Make predictions using the testing set
pred_y = elastic_net_regr.predict(testX)
# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.6f"
      % mean_squared_error(testY, pred_y))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f' % r2_score(testY, pred_y))
