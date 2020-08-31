import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math
from .models import face_photos,predicted_photos
from .forms import input_form


model = load_model('C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\djenv\\face_pro\\face_app\\fmask.h5')
haar = cv2.CascadeClassifier('C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\djenv\\face_pro\\face_app\\haarcascade_frontalface_default.xml')

def predict(image_path,output_path,form):
    label_dict = {0: 'Mask', 1: 'No Mask'}
    color_dict = {1: (0, 0, 255), 0: (0, 255, 0)}
    print(image_path)
    print(output_path)

    image = cv2.imread(image_path)
    # haar = cv2.CascadeClassifier('C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\djenv\\face_pro\\face_app\\haarcascade_frontalface_default.xml')
    # new_image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    faces = haar.detectMultiScale(image)
    for f in faces:
        (x, y, w, h) = [v for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        print(x, y, w, h)
        face_img = image
        resized = cv2.resize(face_img, (224, 224))
        normalized = resized / 255
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        result = model.predict(reshaped)
        print(result)

        # choose the index of the label_dict
        label = np.argmax(result, axis=1)[0]
        print(label)
        text = "{0:s}:{1:.3f}% ".format(label_dict[label], (np.max(result) * 100))

        f_mask = cv2.rectangle(image, (x, y), (x + w, y + h), color_dict[label], 2)
        f_mask = cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_dict[label], 2)
        f_mask = cv2.resize(image, (640, 480))

        #     cv2.imshow('mask',image)
        cv2.imwrite(output_path, f_mask)
    face_photo = predicted_photos()
    face_photo.img_output = 'images_output/'+str(form.cleaned_data['img_input'])
    face_photo.save()
    return face_photo

