# 🛑 Traffic Light Detection using OpenCV

## Description

This project detects traffic light states (Red, Yellow, Green) in **images, videos, and live webcam feeds** using **Python** and **OpenCV**. It uses color segmentation in the HSV color space and contour detection to identify and classify traffic lights in real-time.

---

## Features

* Detect traffic lights in **real-time via webcam**.
* Process **video files** for traffic light detection.
* Test using **static images**.
* Shows bounding boxes around detected traffic lights.
* Displays the current traffic light state (`RED`, `YELLOW`, `GREEN`, or `UNKNOWN`).

---

## Technologies Used

* **Python 3.13**
* **OpenCV** for computer vision
* **NumPy** for numerical operations

---

## Folder Structure

```
Dhriti_RealTime.TrafficLight.Detection/
│
├─ image/                # Sample images for testing
│   ├─ red.jpg
│   ├─ yellow.jpg
│   ├─ green.jpg
│   └─ traffic light.jpg
│
├─ video/                # Sample video for testing
│   └─ traffic signal.mp4
│
├─ traffic_signal_detection.py  # Main Python script
└─ README.md
```

---

## Usage

1. **Clone the repository**:

```bash
git clone https://github.com/ByteBoss-ai/Dhriti_Traffic_Light_Detection.git
```

2. **Install dependencies**:

```bash
pip install opencv-python numpy
```

3. **Run the program**:

```bash
python traffic_signal_detection.py
```

4. **Choose input type**:

```
1 = Webcam
2 = Video file
3 = Image file
```

5. **Follow the prompts**:

* Enter file path for video or image if selected.
* Press `q` to exit live video/webcam feed.

---

## Screenshots

*(Add screenshots here if desired)*

---

## Author

**Dhriti Gupta**
[GitHub Profile](https://github.com/ByteBoss-ai)
