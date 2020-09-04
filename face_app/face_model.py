import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import math
from .models import face_photos,predicted_photos
from .forms import input_form

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR,'face_app/fmask.h5')
model = load_model(model_path)

haar_path = os.path.join(BASE_DIR,'face_app/haarcascade_frontalface_default.xml')
haar = cv2.CascadeClassifier(haar_path)
#haar = cv2.CascadeClassifier('C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\djenv\\face_pro\\face_app\\haarcascade_frontalface_default.xml')

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




### OpenCV Neural network model for improved face detection

prototxt = os.path.join(BASE_DIR,'face_app/model_data/deploy.prototxt')
caffe_model = os.path.join(BASE_DIR,'face_app/model_data/weights.caffemodel')

#prototxt = 'C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\djenv\\face_pro\\face_app\\model_data\\deploy.prototxt'
#caffe_model = 'C:\\Users\\prajv\\Desktop\\PycharmProjects\\PS-PY\\venv\\djenv\\face_pro\\face_app\\model_data\\weights.caffemodel'
face_detect_model = cv2.dnn.readNetFromCaffe(prototxt,caffe_model)

def predict_v2(image_path,output_path,form):
    label_dict = {0: 'Mask', 1: 'No Mask'}
    color_dict = {1: (0, 0, 255), 0: (0, 255, 0)}

    raw_image = cv2.imread(image_path)
    image = resized = cv2.resize(raw_image, (1024, 768))
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detect_model.setInput(blob)
    detections = face_detect_model.forward()
    # image= cv2.resize(raw_image,(224,224))
    (h, w) = image.shape[:2]
    count = 0
    with_mask = 0
    without_mask = 0
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.4):
            face_img = image[startY:endY, startX:endX]
            resized = cv2.resize(face_img, (224, 224))
            normalized = resized / 255
            reshaped = np.reshape(normalized, (1, 224, 224, 3))
            result = model.predict(reshaped)
            print(result)

            # choose the index of the label_dict
            label = np.argmax(result, axis=1)[0]
            print(label)
            if label == 0:
                with_mask = with_mask+1
            else:
                without_mask = without_mask+1

            text = "{0:s}:{1:.3f}% ".format(label_dict[label], (np.max(result) * 100))
            f_mask = cv2.rectangle(image, (startX, startY), (endX, endY), color_dict[label], 2)
            f_mask = cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_dict[label],2)
            f_mask = cv2.resize(image, (640, 480))
            count=count+1

        cv2.imwrite(output_path, f_mask)
    face_photo = predicted_photos()
    face_photo.img_output = 'images_output/' + str(form.cleaned_data['img_input'])
    face_photo.save()
    return face_photo,count,with_mask,without_mask
