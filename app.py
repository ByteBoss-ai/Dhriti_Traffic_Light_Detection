import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av  # <-- needed for converting frames

st.title("🚦 Real-Time Traffic Light Detection")

# HSV Ranges
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
lower_yellow, upper_yellow = np.array([15, 100, 100]), np.array([35, 255, 255])
lower_green, upper_green = np.array([40, 100, 100]), np.array([90, 255, 255])

def draw_contours(mask, color, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def classify_state(mask_red, mask_yellow, mask_green):
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)
    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        return "RED"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        return "YELLOW"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        return "GREEN"
    else:
        return "UNKNOWN"

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    draw_contours(mask_red, (0, 0, 255), frame)
    draw_contours(mask_yellow, (0, 255, 255), frame)
    draw_contours(mask_green, (0, 255, 0), frame)

    state = classify_state(mask_red, mask_yellow, mask_green)
    cv2.putText(frame, f"Traffic Light: {state}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# FIXED VideoTransformer
class VideoTransformer(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = process_frame(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")  # FIXED

# Sidebar
st.sidebar.title("Options")
input_type = st.sidebar.radio("Select input type:", ("Webcam", "Image Upload", "Video Upload"))

if input_type == "Webcam":
    webrtc_streamer(
        key="example",
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )

elif input_type == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        processed = process_frame(img)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")

elif input_type == "Video Upload":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame)
            stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()
