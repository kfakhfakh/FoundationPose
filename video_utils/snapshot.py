import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

CHESSBOARD_SIZE = (8, 5)

CALIB_FILE = "realsense_calib.npz"

# ---------------- Load calibration if exists ----------------
calibrated = False
camera_matrix = None
dist_coeffs = None

if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    calibrated = True

    print("Calibration loaded!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coeffs:\n", dist_coeffs)
else:
    print("No calibration found → running raw stream")

# ---------------- RealSense setup ----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("SPACE = snapshot | ESC = exit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # ---------------- Apply calibration if available ----------------
        if calibrated:
            h, w = frame.shape[:2]
            newK, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newK)

        display = frame.copy()

        # Chessboard detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if found:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)

            # vertical line
            top = tuple(corners[0][0].astype(int))
            bottom = tuple(corners[CHESSBOARD_SIZE[0] * (CHESSBOARD_SIZE[1] - 1)][0].astype(int))
            cv2.line(display, top, bottom, (0, 0, 255), 2)

        cv2.imshow("RealSense View (Calibrated if available)", display)

        key = cv2.waitKey(1) & 0xFF

        # SPACE → snapshot + matplotlib
        if key == 32:
            print("Snapshot captured")

            snapshot_rgb = display[:, :, ::-1]

            plt.figure(figsize=(8, 6))
            plt.imshow(snapshot_rgb)
            plt.title("Snapshot")
            plt.axis("off")
            plt.show()

        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
