import numpy as np
from data_preprocessing import load_ucf101_dataset
from feature_extractor import build_feature_extractor, extract_features
from model import build_lstm_model

def train_model():
    print("Loading dataset...")
    X, y = load_ucf101_dataset('datasets/ucf101')

    print("Extracting features...")
    feature_model = build_feature_extractor()
    X_features = [extract_features(feature_model, x) for x in X]
    X_features = np.array(X_features)

    print("Training LSTM model...")
    model = build_lstm_model(input_shape=(X_features.shape[1], X_features.shape[2]), num_classes=len(set(y)))
    model.fit(X_features, y, epochs=10, batch_size=8)
    model.save('action_recognition_model.h5')
    print("Model saved as action_recognition_model.h5")

if __name__ == '__main__':
    train_model()