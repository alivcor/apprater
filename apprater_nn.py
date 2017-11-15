import cv2, numpy as np
from vgg16_feat import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten, Dense, Input
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D


vgg_model = VGG16(include_top=True, weights='imagenet')

vgg_feature_vector = vgg_model.layers[-2].output
# print vgg_feature_vector.shape

feature_extraction_model = Model(input=[vgg_model.input], output=[vgg_feature_vector])
feature_extraction_model.summary()


cap = cv2.VideoCapture('dataset/dataset_100/1/Modern Combat 5 - Game Trailer-_jSPJWqQM90.mp4')
# # Read the first frame of the video
ret, frame = cap.read()
print frame.shape
im = cv2.resize(frame, (224, 224)).astype(np.float32)
im = np.expand_dims(im, axis=0)
print im.shape
print feature_extraction_model.predict(im).shape

#
# apprater_model = Sequential()
# apprater_model.add(Dense(12, activation='relu', name='rating'))
# apprater_model.add(Dense(8, activation='tanh'))
# apprater_model.add(Dense(8, activation='relu'))
# apprater_model.add(Dense(5, activation='softmax'))
# apprater_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# apprater_model.summary()

# x = Dense(8, activation='softmax', name='predictions')(vgg_model.layers[-2].output)
# app_rater_cnn = Model(input=vgg_model.input, output=x)
# app_rater_cnn.summary()

#
# cap = cv2.VideoCapture('dataset/dataset_100/1/Modern Combat 5 - Game Trailer-_jSPJWqQM90.mp4')
# # Read the first frame of the video
# ret, frame = cap.read()
# print frame.shape
#
# apprater_model.fit(train_X, train_Y, epochs=200, batch_size=10)


