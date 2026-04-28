import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Chessboard config
CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE = 2.85

# Prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

frame_count = 0
calibrated = False
camera_matrix = None
dist_coeffs = None

# === RealSense setup ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

print("Controls:")
print("SPACE → capture")
print("C → calibrate")
print("R → reset")
print("S → save")
print("L → load")
print("ESC → exit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect chessboard
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if found:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)

            # ✅ VERTICAL LINE (instead of diagonal)
            top_corner = tuple(corners[0][0].astype(int))
            bottom_corner = tuple(corners[CHESSBOARD_SIZE[0] * (CHESSBOARD_SIZE[1] - 1)][0].astype(int))

            cv2.line(display, top_corner, bottom_corner, (0, 0, 255), 2)

        # Frame counter
        cv2.putText(display, f"Frames: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show stream
        if calibrated:
            h, w = frame.shape[:2]

            newK, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )

            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newK)

            combined = np.hstack((frame, undistorted))
            cv2.imshow("Original | Undistorted", combined)
        else:
            cv2.imshow("RealSense RGB", display)

        key = cv2.waitKey(1) & 0xFF

        # SPACE → capture
        if key == ord(' '):
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
                frame_count += 1
                print(f"Captured frame {frame_count}")
            else:
                print("Chessboard not detected")

        # C → calibrate
        elif key == ord('c'):
            if len(objpoints) > 0:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )

                calibrated = True

                print("\n=== CALIBRATION DONE ===")
                print("Camera Matrix:\n", camera_matrix)
                print("Distortion Coeffs:\n", dist_coeffs)

            else:
                print("No frames!")

        # R → reset
        elif key == ord('r'):
            objpoints = []
            imgpoints = []
            frame_count = 0
            calibrated = False
            print("Reset done")

        # S → save
        elif key == ord('s'):
            if calibrated:
                np.savez("realsense_calib.npz",
                         camera_matrix=camera_matrix,
                         dist_coeffs=dist_coeffs)

                np.savetxt("cam_K.txt", camera_matrix)

                print("\nSaved calibration!")
                print("K:\n", camera_matrix)
                print("Distortion:\n", dist_coeffs)
            else:
                print("Nothing to save")

        # L → load
        elif key == ord('l'):
            if os.path.exists("realsense_calib.npz"):
                data = np.load("realsense_calib.npz")

                camera_matrix = data["camera_matrix"]
                dist_coeffs = data["dist_coeffs"]

                calibrated = True

                print("\n=== LOADED CALIBRATION ===")
                print("Camera Matrix:\n", camera_matrix)
                print("Distortion Coeffs:\n", dist_coeffs)

                print("\nFull values breakdown:")
                print(f"fx = {camera_matrix[0,0]}")
                print(f"fy = {camera_matrix[1,1]}")
                print(f"cx = {camera_matrix[0,2]}")
                print(f"cy = {camera_matrix[1,2]}")
                print("dist =", dist_coeffs.ravel())

            else:
                print("File not found")

        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
