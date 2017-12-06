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
from scipy import stats
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
    graphics_model.add(Dense(8, activation="relu"))
    graphics_model.add(Dense(1, kernel_initializer='normal', kernel_constraint=min_max_norm(min_value=0.0, max_value=5.0), kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    graphics_model.compile(loss='mean_squared_error', optimizer=adam)
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
    filtered_indices = []
    cnt = 0
    for x in alldata_Y:
        if x != 0:
            filtered_indices.append(cnt)
        cnt+=1
    # indices = np.arange(alldata_X.shape[0]-1)
    indices = filtered_indices
    print "Number of data points filtered: ", len(indices)
    np.random.shuffle(indices)

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
    feature_vectors = load_obj("feature_vectors_complete_v2")
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
    mu = np.mean(x)
    sigma = np.std(x)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth = 2, color = 'r')
    plt.show()
    # sys.exit(0)
    pass


def analyze_data(trainY, testY):
    ratings = trainY
    plot_hist(ratings)
    print stats.describe(ratings)
    monkey_preds = np.random.normal(np.mean(trainY), np.std(trainY), testX.shape[0])
    print("Mean squared error: %.6f"
          % mean_squared_error(testY, monkey_preds))
    print('Variance score: %.6f' % r2_score(testY, monkey_preds))
    pass


def bin_count(trainY):
    bin1 = 0
    bin2 = 0
    bin3 = 0
    bin4 = 0
    bin5 = 0
    for x in trainY:
        if x < 1:
            bin1 += 1
        elif x >= 1 and x < 2:
            bin2 += 1
        elif x >= 2 and x < 3:
            bin3 += 1
        elif x >= 3 and x < 4:
            bin4 += 1
        else:
            bin5 += 1
    print bin1, bin2, bin3, bin4, bin5

trainX, trainY, testX, testY, gtrainX, gtestX = loadDataset()

textual_only = False
trivial_only = False

analyze_data(trainY, testY)

print np.amax(gtrainX[10,:,:].flatten()), np.amin(gtrainX[10,:,:].flatten())


# # MARK: GRAPHICS MODEL TRAINING
graphics_model = compileGraphicsModel()
graphics_model.fit(gtrainX, trainY, batch_size=12, epochs=250)
graphics_model.save("obj/trained_graphic_model.h5")


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
apprater_model = compileMainModel()
apprater_model.fit(trainX, trainY, batch_size=32, epochs=1000)
apprater_model.save("obj/apprater_model.h5")
print "\n\nNeural Network:\n"
print apprater_model.evaluate(x=testX, y=testY)
# print "\n"
# print "Predicted Output: ", apprater_model.predict(trainX[0,:].reshape(1, -1))




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


#
#
if textual_only:
    trainX = trainX[:,7]
    testX = testX[:,7]
    print trainX[7]
    trainX2 = []
    testX2 = []

    for x in trainX:
        trainX2.append([x])
    trainX = np.array(trainX2)

    for x in testX:
        testX2.append([x])
    testX = np.array(testX2)

if trivial_only:
    trainX = np.delete(trainX, 7, 1)
    testX = np.delete(testX, 7, 1)

print trainX.shape
print "\n\nElastic Net Regression: \n"
elastic_net_regr = ElasticNet(random_state=2)
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

