# import cv2
# import face_recognition

# # OpenCV loads in BGR, convert to RGB
# image = cv2.imread("test.jpg")
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# face_locations = face_recognition.face_locations(rgb_image)
# print("Found {} face(s)".format(len(face_locations)))

from keras_facenet import FaceNet
embedder = FaceNet()
print(embedder)
