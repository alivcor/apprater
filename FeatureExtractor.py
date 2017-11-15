import cv2, numpy as np
from vgg16_feat import VGG16
from keras.models import Model


def _preprocess_input(input_tensor):
    print "Preprocessing Input Frame..."
    processed_input_tensor = cv2.resize(input_tensor, (224, 224)).astype(np.float32)
    processed_input_tensor = np.expand_dims(processed_input_tensor, axis=0)
    print "Preprocessing Complete !"
    print "processed_input_tensor.shape : ", processed_input_tensor.shape
    return processed_input_tensor


def init_load_extractor_model():
    print "Loading Extractor Model.."

    vgg_model = VGG16(include_top=True, weights='imagenet')

    print "Shredding Softmax Layer.."
    vgg_feature_vector = vgg_model.layers[-2].output
    # print vgg_feature_vector.shape

    feature_extraction_model = Model(input=[vgg_model.input], output=[vgg_feature_vector])
    feature_extraction_model.summary()
    return feature_extraction_model


def extract_features(model, input_tensors):
    feature_vectors = []

    print "Extracting Features.."

    for input_tensor in input_tensors:
        processed_input_tensor = _preprocess_input(input_tensor)
        feature_vector = model.predict(processed_input_tensor)
        feature_vectors.append(feature_vector)

    print "Features Extracted !"
    return np.array(feature_vectors)