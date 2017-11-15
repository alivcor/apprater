import cv2, numpy as np
import sys, time, datetime
import FeatureExtractor, EventIssuer
import progressbar
from keras.layers import Flatten, Dense, Input
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
_LOGFILENAME = ""
reload(sys)


def start_iresium_core():
    global _LOGFILENAME, timestamp
    timestamp = time.time()
    strstamp = datetime.datetime.fromtimestamp(timestamp).strftime('%m-%d-%Y %H:%M:%S')
    _LOGFILENAME = "logs/Iresium_Log_" + str(timestamp)
    np.random.seed(7)
    EventIssuer.issueWelcome(_LOGFILENAME)
    EventIssuer.genLogFile(_LOGFILENAME, timestamp, strstamp)
    return _LOGFILENAME, timestamp


def compileMainModel():
    apprater_model = Sequential()
    apprater_model.add(Dense(12, activation='relu', name='rating'))
    apprater_model.add(Dense(8, activation='tanh'))
    apprater_model.add(Dense(8, activation='relu'))
    apprater_model.add(Dense(5, activation='softmax'))
    apprater_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    apprater_model.summary()


def process_videos(logfilename):
    filename = 'dataset/dataset_100/1/Modern Combat 5 - Game Trailer-_jSPJWqQM90.mp4'
    feature_extraction_model = FeatureExtractor.init_load_extractor_model(_LOGFILENAME)
    vgg_features = FeatureExtractor.extract_features(filename, feature_extraction_model, _LOGFILENAME, stride=50)

    # for i in range(1, 101):


start_iresium_core()





#
