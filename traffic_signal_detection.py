import cv2
import numpy as np

# 1. Choosing Input Source
print("Choose input type:")
print("1 = Webcam")
print("2 = Video file")
print("3 = Image file")
choice = input("Enter choice (1/2/3): ")

if choice == "1":
    cap = cv2.VideoCapture(0)   # Webcam
    mode = "webcam"
elif choice == "2":
    file_path = input("Enter video file path: ")
    cap = cv2.VideoCapture(file_path)

    mode = "video"
elif choice == "3":
    file_path = input("Enter image file path: ")
    img = cv2.imread(file_path)
    mode = "image"
else:
    print("Invalid choice, defaulting to webcam...")
    cap = cv2.VideoCapture(0)
    mode = "webcam"


# 2. HSV Ranges for colors
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
lower_yellow, upper_yellow = np.array([15, 100, 100]), np.array([35, 255, 255])
lower_green, upper_green = np.array([40, 100, 100]), np.array([90, 255, 255])


# 3. Function to draw contours
def draw_contours(mask, color, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # ignore tiny spots
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


# 4. Function to classify traffic light state
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


# New function to process a frame (common code)
def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Draw bounding boxes
    draw_contours(mask_red, (0, 0, 255), frame)
    draw_contours(mask_yellow, (0, 255, 255), frame)
    draw_contours(mask_green, (0, 255, 0), frame)

    # Classify
    state = classify_state(mask_red, mask_yellow, mask_green)

    # Show result
    cv2.putText(frame, f"Traffic Light: {state}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame


# 5. Image Mode
if mode == "image":
    frame = img.copy()
    processed_frame = process_frame(frame)
    cv2.imshow("Traffic Light Detection", processed_frame)
    cv2.waitKey(0)  # wait for a key press
    cv2.destroyAllWindows()


# 6. Webcam / Video Mode
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Traffic Light Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()