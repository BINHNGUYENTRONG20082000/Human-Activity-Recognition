import tensorflow as tf
from tensorflow import keras
import cv2
from poseDetector import poseDetector
import numpy as np
from predictVideo import predict_on_video
from moviepy.editor import VideoFileClip

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["fight", "Trespassing","stroke"]

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 128,128

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20
new_model = tf.keras.models.load_model('models/LRCN_with_LSTM(dropout0.25).h5')
input_video_file_path = "Data/stroke/5.mp4"

# Construct the output video path.
output_video_file_path = 'Output-SeqLen10.mp4'

# Perform Action Recognition on the Test Video.
predict_on_video(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)

# Display the output video.
VideoFileClip(output_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()



