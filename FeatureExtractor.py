import cv2, numpy as np
from vgg16_feat import VGG16
from keras.models import Model


def _preprocess_input(input_tensors):
    print "Preprocessing Input Frames..."
    processed_input_tensors = []

    for input_tensor in input_tensors:
        im = cv2.resize(input_tensor, (224, 224)).astype(np.float32)
        im = np.expand_dims(im, axis=0)
        processed_input_tensors.append(im)

    print "Preprocessing Complete !"
    processed_input_tensors = np.array(processed_input_tensors)
    print "processed_input_tensors.shape : ", processed_input_tensors.shape
    return processed_input_tensors


def load_extractor_model():
    vgg_model = VGG16(include_top=True, weights='imagenet')
    vgg_feature_vector = vgg_model.layers[-2].output
    # print vgg_feature_vector.shape

    feature_extraction_model = Model(input=[vgg_model.input], output=[vgg_feature_vector])
    feature_extraction_model.summary()
    return feature_extraction_model


def extract_features(model, input_tensors):
    input_tensors = _preprocess_input(input_tensors)

    print "Extracting Features.."
    feature_vectors = model.predict(input_tensors)

    print "Features Extracted !"
    return feature_vectors