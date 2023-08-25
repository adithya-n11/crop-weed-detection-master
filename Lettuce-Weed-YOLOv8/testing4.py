import cv2
from ultralytics import YOLO
from keras.models import load_model
import numpy as np
import streamlit as st
import time
from moviepy.editor import *

yolo_model = YOLO("yolov8m.pt")

distraction_model = load_model("models/vgg19_small.h5")

distraction_classes = ['Safe Driving', 'Operating the radio',
                       'Drinking', 'Reaching behind', 'Hair and makeup', 'Talking to passenger']

def speed_up_video(video_path):
    clip = VideoFileClip(video_path)
    final = clip.fx( vfx.speedx, 10)
    final.ipython_display()

def process_video(video_path, video_placeholder):

    speed_up_video(video_path)

    cap = cv2.VideoCapture('__temp__.mp4')

    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        process_frame(frame)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        display_frame(frame, video_placeholder)

        time.sleep(0.001)

    cap.release()

def process_webcam(video_placeholder):
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        process_frame(frame)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        display_frame(frame, video_placeholder)

        time.sleep(0)

    cap.release()

def process_frame(frame):
    yolo_results = yolo_model.predict(frame)
    yolo_result = yolo_results[0]

    has_cell_phone = False

    for box in yolo_result.boxes:
        class_id = yolo_result.names[box.cls[0].item()]
        if class_id == "cell phone":
            has_cell_phone = True
            cv2.putText(frame, "Using Cell Phone", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            break

    if not has_cell_phone:
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        expanded_frame = np.expand_dims(normalized_frame, axis=0)

        predictions = distraction_model.predict(expanded_frame)
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = distraction_classes[predicted_class_index]
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def display_frame(frame, video_placeholder):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

st.set_page_config(page_title='Driver Distraction Detection')

st.title('Driver Distraction Detection')
selected_option = st.sidebar.selectbox("Select Input", ("Video", "Webcam"))

if selected_option == "Video":
    uploaded_file = st.file_uploader('Upload a video', type=['mp4', 'avi', 'mpeg'])

    if uploaded_file is not None:
        video_path = "uploaded_video.mp4"
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        video_placeholder = st.empty()

        process_video(video_path, video_placeholder)

elif selected_option == "Webcam":
    video_placeholder = st.empty()

    process_webcam(video_placeholder)
