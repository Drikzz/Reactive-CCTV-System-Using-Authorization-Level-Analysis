from utils.face_classifier import load_known_faces, train_and_save_svm_classifier

faces, names = load_known_faces("datasets/faces")  # or path in your Colab files
train_and_save_svm_classifier(faces, names)