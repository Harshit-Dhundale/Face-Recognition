# Importing Required Libraries:

import face_recognition as fr
import cv2
import numpy as np
import os

# Variables Used:

path = "C:/Users/harsh/Desktop/Project Review 287/face-recognition-python-code/train/"
path1 = "D:/MEDIA/img_align_celeba/img_align_celeba/"
known_names = []
known_name_encodings = []

# Training AI:

for x in os.listdir(path):
    image = fr.load_image_file(path + x)
    image_path = path + x
    encoding = fr.face_encodings(image)[0]
    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

# Testing the AI:

file_path = input("Enter the File Path of the Test Image:" )
lenght = len(file_path)
test_image = file_path[1:lenght-1]

# Recognizing Faces:

image = cv2.imread(test_image)
face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
   matches = fr.compare_faces(known_name_encodings, face_encoding)
   name = ""

   face_distances = fr.face_distance(known_name_encodings, face_encoding)
   best_match = np.argmin(face_distances)

   if matches[best_match]:
       name = known_names[best_match]

   cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
   cv2.rectangle(image, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)

   font = cv2.FONT_HERSHEY_DUPLEX
   cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (55,255,20), 1)


cv2.imshow("Face Recognition", image)
cv2.imwrite("./output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()