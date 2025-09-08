from flask import Flask, request, Response, send_file, render_template_string, jsonify
import cv2
import numpy as np
import io
import threading

app = Flask(__name__)

# ================= Traffic Light Detection Logic =================
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
lower_yellow, upper_yellow = np.array([15, 100, 100]), np.array([35, 255, 255])
lower_green, upper_green = np.array([40, 100, 100]), np.array([90, 255, 255])

def draw_contours(mask, color, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
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
    return "UNKNOWN"

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    draw_contours(mask_red, (0, 0, 255), frame)
    draw_contours(mask_yellow, (0, 255, 255), frame)
    draw_contours(mask_green, (0, 255, 0), frame)

    state = classify_state(mask_red, mask_yellow, mask_green)
    cv2.putText(frame, f"Traffic Light: {state}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# ================= Webcam Handling =================
camera = None
webcam_active = False
lock = threading.Lock()

def generate_webcam():
    global camera, webcam_active
    while webcam_active and camera is not None:
        ret, frame = camera.read()
        if not ret:
            break
        processed = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global camera, webcam_active
    with lock:
        if not webcam_active:
            camera = cv2.VideoCapture(0)
            webcam_active = True
    return jsonify({"status": "started"})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global camera, webcam_active
    with lock:
        webcam_active = False
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"status": "stopped"})

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= Flask Routes =================
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚦 Traffic Light Detection</title>
        <style>
            body {
                margin: 0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #1e1e2f;
                color: #f0f0f0;
                text-align: center;
            }
            header {
                background: linear-gradient(90deg, #007bff, #00d4ff);
                padding: 20px;
                font-size: 24px;
                font-weight: bold;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            .container {
                max-width: 1000px;
                margin: 40px auto;
                padding: 20px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }
            .card {
                background: #2a2a40;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 6px 10px rgba(0,0,0,0.4);
            }
            h2 {
                margin-bottom: 15px;
                color: #00d4ff;
            }
            input[type=file] {
                margin: 10px 0;
                color: #ccc;
            }
            button, .btn {
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: 0.3s;
                text-decoration: none;
                display: inline-block;
            }
            button:hover, .btn:hover {
                background: #0056b3;
            }
            .video-box {
                margin-top: 15px;
                border-radius: 10px;
                overflow: hidden;
                border: 2px solid #00d4ff;
                display: flex;
                justify-content: center;
                align-items: center;
                max-width: 800px;
                margin-left: auto;
                margin-right: auto;
                aspect-ratio: 16 / 9;
                background: black;
            }
            #webcam {
                width: 100%;
                height: 100%;
                border-radius: 10px;
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <header> Traffic Light Detection System</header>
        <div class="container">
            
            <!-- Upload Image -->
            <div class="card">
                <h2>Upload Image</h2>
                <form action="/process_image" method="post" enctype="multipart/form-data" target="_blank">
                    <input type="file" name="file" required>
                    <button type="submit">Upload & Process</button>
                </form>
                <p>Processed image will open in a new tab.</p>
            </div>
            
            <!-- Upload Video -->
            <div class="card">
                <h2>Upload Video</h2>
                <form action="/process_video" method="post" enctype="multipart/form-data" target="_blank">
                    <input type="file" name="file" required>
                    <button type="submit">Upload & Stream</button>
                </form>
            </div>
            
            <!-- Webcam -->
            <div class="card" style="grid-column: span 2;">
                <h2>Live Webcam</h2>
                <div>
                    <button onclick="startWebcam()">Start Webcam</button>
                    <button onclick="stopWebcam()">Stop Webcam</button>
                </div>
                <div class="video-box">
                    <img id="webcam" src="" alt="Webcam Stream">
                </div>
            </div>
            
        </div>
        
        <script>
            function startWebcam() {
                fetch("/start_webcam", {method: "POST"}).then(() => {
                    document.getElementById("webcam").src = "/webcam_feed";
                });
            }
            function stopWebcam() {
                fetch("/stop_webcam", {method: "POST"}).then(() => {
                    document.getElementById("webcam").src = "";
                });
            }
        </script>
    </body>
    </html>
    """)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file selected", 400
    file = request.files['file']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    processed = process_frame(img)
    _, buffer = cv2.imencode(".png", processed)
    return send_file(io.BytesIO(buffer), mimetype='image/png')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file selected", 400
    file_path = "temp_video.mp4"
    request.files['file'].save(file_path)
    cap = cv2.VideoCapture(file_path)

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # use the PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port)

