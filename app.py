import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.title("Real-Time Traffic Light Detection")

# 1. Choose input type
input_type = st.radio("Select input type:", ("Webcam", "Video file", "Image file"))

# 2. HSV Ranges
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
lower_yellow, upper_yellow = np.array([15, 100, 100]), np.array([35, 255, 255])
lower_green, upper_green = np.array([40, 100, 100]), np.array([90, 255, 255])

# 3. Functions
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
    draw_contours(mask_red, (0,0,255), frame)
    draw_contours(mask_yellow, (0,255,255), frame)
    draw_contours(mask_green, (0,255,0), frame)
    state = classify_state(mask_red, mask_yellow, mask_green)
    cv2.putText(frame, f"Traffic Light: {state}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return frame

# 4. Handle Input
if input_type == "Image file":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_file.read())
        tfile.close()
        img = cv2.imread(tfile.name)
        os.remove(tfile.name)  # safe to delete after reading
        processed = process_frame(img)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

elif input_type == "Video file":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame)
            stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        cap.release()
        os.remove(tfile.name)  # delete after release

elif input_type == "Webcam":
    st.warning("Webcam streaming is not supported directly in Streamlit. Please upload a video instead.")
