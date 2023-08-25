import cv2
from ultralytics import YOLO
import streamlit as st
import time
# Load the YOLOv8 model
model = YOLO('best.pt')



def process_video(video_path, video_placeholder):

    cap = cv2.VideoCapture(video_path)

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

        # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
    yolo_results = model.predict(frame)
    yolo_result = yolo_results[0]

    has_cell_phone = False

    for box in yolo_result.boxes:
        class_id = yolo_result.names[box.cls[0].item()]
        if class_id == "crop":
            cv2.putText(frame, "Lettuce", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            break
        if class_id == "ca" or "cs" or "bg":
            cv2.putText(frame, "Weed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            break

def display_frame(frame, video_placeholder):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)


st.set_page_config(page_title='Lettuce Crop Weed Prediction')

st.title('Lettuce Crop Weed Prediction')
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


