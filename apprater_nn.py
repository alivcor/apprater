import cv2, numpy as np
import sys, time, datetime
import FeatureExtractor, EventIssuer
import progressbar
from keras.layers import Flatten, Dense, Input
from keras.models import Sequential
import glob, os, pickle
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



def save_obj(obj, name, logfilename):
    EventIssuer.issueSharpAlert("Saving data ..", logfilename, True)
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    EventIssuer.issueSuccess("Saved Object ..", logfilename, True)


def load_obj(name, logfilename):
    EventIssuer.issueSharpAlert("Loading data ..", logfilename, True)
    with open('obj/' + name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
        EventIssuer.issueSuccess("Loaded Object ..", logfilename, True)
        return obj


def compileMainModel():
    apprater_model = Sequential()
    apprater_model.add(Dense(12, activation='relu', name='rating'))
    apprater_model.add(Dense(8, activation='tanh'))
    apprater_model.add(Dense(8, activation='relu'))
    apprater_model.add(Dense(5, activation='softmax'))
    apprater_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    apprater_model.summary()


def process_videos(dir_path, features_dict, logfilename):
    first_app_incl = 1096 # was 101
    last_app_incl = 1494 # was 1095
    feature_extraction_model = FeatureExtractor.init_load_extractor_model(_LOGFILENAME)
    for i in range(first_app_incl, last_app_incl+1):
        dir_name = dir_path + str(i) + "/"
        files = glob.glob(os.path.join(dir_name, '*.mp4'))
        EventIssuer.issueMessage("Processing " + str(i) + " of " + str(last_app_incl-first_app_incl+1), logfilename)
        if(files):
            EventIssuer.issueSuccess("Video files found - extracting features.", logfilename)
            filename = files[0]
            vgg_features = FeatureExtractor.extract_features(filename, feature_extraction_model, _LOGFILENAME, stride=10, max_frames=500)
            features_dict[i] = vgg_features
        else:
            EventIssuer.issueWarning("No videos found, setting FV to 0", logfilename)
            features_dict[i] = 0
        save_obj(features_dict, "feature_vectors_complete_v2", logfilename)
        EventIssuer.issueSuccess("Processed " + str(i) + " of " + str(last_app_incl-first_app_incl+1), logfilename, True)


start_iresium_core()
features_dict = load_obj("feature_vectors_complete", _LOGFILENAME)
process_videos("/clean_dataset/",features_dict, _LOGFILENAME)
print len(features_dict)

