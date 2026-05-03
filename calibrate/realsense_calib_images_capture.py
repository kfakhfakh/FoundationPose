import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# -------------------------
# Settings
# -------------------------
CHECKERBOARD = (8, 5)  # inner corners (change if needed)
SAVE_DIR = "calibration_images"

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# RealSense setup
# -------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

pipeline.start(config)

# -------------------------
# Helper
# -------------------------
def save_image(image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(SAVE_DIR, f"calib_{timestamp}.png")
    cv2.imwrite(filename, image)
    print(f"[SAVED] {filename}")

# -------------------------
# Main loop
# -------------------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale for detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect chessboard
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = color_image.copy()

        if found:
            # refine corners for better accuracy
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

            cv2.putText(display, "Chessboard detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(display, "No chessboard", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Calibration Capture", cv2.resize(display, (640, 480)))

        key = cv2.waitKey(1) & 0xFF

        # -------------------------
        # Press 'c' to save RAW frame
        # -------------------------
        if key == ord('c'):
            save_image(color_image)

        # ESC to exit
        if key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
