import sys
import os

# Add the root project dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.face_classifier import load_known_faces, train_and_save_svm_classifier

faces, names = load_known_faces("datasets/faces")  # or path in your Colab files
train_and_save_svm_classifier(faces, names)