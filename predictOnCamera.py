import numpy as np
import cv2
import tensorflow as tf
import os
from collections import deque
from poseDetector import poseDetector

# Load mô hình
model_path = 'models/LRCN_with_LSTM(dropout0.25).h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file does not exist: {model_path}")

LRCN_model = tf.keras.models.load_model(model_path)
LRCN_model.summary()

IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["fight", "Trespassing", "stroke"]

def predict_from_camera(SEQUENCE_LENGTH):
    # Initialize the VideoCapture object to read from the webcam.
    video_reader = cv2.VideoCapture(0)
    if not video_reader.isOpened():
        raise ValueError("Could not open webcam")

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    while True:
        ok, frame = video_reader.read()
        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        processed_frame = poseDetector(resized_frame)
        frames_queue.append(processed_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_reader.release()
    cv2.destroyAllWindows()

# Gọi hàm để bắt đầu dự đoán từ camera
predict_from_camera(SEQUENCE_LENGTH)
