from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import numpy as np

def build_feature_extractor():
    base_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg')
    return base_model

def extract_features(model, frames):
    frames = frames.astype('float32') / 255.0
    return np.array([model.predict(np.expand_dims(frame, axis=0))[0] for frame in frames])
