import cv2, numpy as np
import FeatureExtractor
from keras.layers import Flatten, Dense, Input
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D





cap = cv2.VideoCapture('dataset/dataset_100/1/Modern Combat 5 - Game Trailer-_jSPJWqQM90.mp4')
# # Read the first frame of the video
ret, frame = cap.read()
print frame.shape

vgg_features = FeatureExtractor.extract_features(frame)

apprater_model = Sequential()
apprater_model.add(Dense(12, activation='relu', name='rating'))
apprater_model.add(Dense(8, activation='tanh'))
apprater_model.add(Dense(8, activation='relu'))
apprater_model.add(Dense(5, activation='softmax'))
apprater_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
apprater_model.summary()

