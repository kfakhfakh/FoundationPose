import pyrealsense2 as rs
import numpy as np
import cv2
import os

# =========================
# Chessboard config
# =========================
CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE = 2.85

# Prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

frame_count = 0
calibrated = False
camera_matrix = None
dist_coeffs = None


# =========================
# 📊 Reprojection error (calibration case)
# =========================
def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist
        )

        img = imgpoints[i].reshape(-1, 2)
        proj = projected.reshape(-1, 2)

        error = np.linalg.norm(img - proj)

        total_error += error
        total_points += len(img)

    return total_error / total_points


# =========================
# 📊 Evaluate LOADED calibration (solvePnP)
# =========================
def evaluate_loaded_calibration(objpoints, imgpoints, K, dist):
    total_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        success, rvec, tvec = cv2.solvePnP(
            objpoints[i],
            imgpoints[i],
            K,
            dist
        )

        if not success:
            continue

        projected, _ = cv2.projectPoints(
            objpoints[i], rvec, tvec, K, dist
        )

        img = imgpoints[i].reshape(-1, 2)
        proj = projected.reshape(-1, 2)

        error = np.linalg.norm(img - proj)

        total_error += error
        total_points += len(img)

    if total_points == 0:
        return -1

    return total_error / total_points


# =========================
# RealSense setup
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("Controls:")
print("SPACE → capture")
print("C → calibrate")
print("L → load calibration")
print("E → evaluate loaded calibration")
print("R → reset frames")
print("S → save")
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

        cv2.putText(display, f"Frames: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show stream
        if calibrated:
            h, w = frame.shape[:2]

            newK, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )

            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newK)
            cv2.imshow("Original | Undistorted", np.hstack((frame, undistorted)))
        else:
            cv2.imshow("RealSense RGB", display)

        key = cv2.waitKey(1) & 0xFF

        # =========================
        # SPACE → capture
        # =========================
        if key == ord(' '):
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
                frame_count += 1
                print(f"Captured frame {frame_count}")
            else:
                print("Chessboard not detected")

        # =========================
        # C → calibrate
        # =========================
        elif key == ord('c'):
            if len(objpoints) > 0:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )

                calibrated = True

                print("\n=== CALIBRATION DONE ===")
                print("Camera Matrix:\n", camera_matrix)
                print("Distortion Coeffs:\n", dist_coeffs)

                error = compute_reprojection_error(
                    objpoints, imgpoints, rvecs, tvecs,
                    camera_matrix, dist_coeffs
                )

                print(f"\n📊 Calibration reprojection error: {error:.4f} px")

            else:
                print("No frames captured")

        # =========================
        # L → load calibration
        # =========================
        elif key == ord('l'):
            if os.path.exists("realsense_calib.npz"):
                data = np.load("realsense_calib.npz")

                camera_matrix = data["camera_matrix"]
                dist_coeffs = data["dist_coeffs"]

                calibrated = True

                print("\n=== LOADED CALIBRATION ===")
                print("Camera Matrix:\n", camera_matrix)
                print("Distortion Coeffs:\n", dist_coeffs)

            else:
                print("File not found")

        # =========================
        # E → evaluate loaded calibration
        # =========================
        elif key == ord('e'):
            if calibrated and len(objpoints) > 0:
                error = evaluate_loaded_calibration(
                    objpoints,
                    imgpoints,
                    camera_matrix,
                    dist_coeffs
                )

                if error >= 0:
                    print(f"\n📊 Loaded calibration reprojection error: {error:.4f} px")
                else:
                    print("Evaluation failed")
            else:
                print("Need calibration + captured frames")

        # =========================
        # S → save
        # =========================
        elif key == ord('s'):
            if calibrated:
                np.savez("realsense_calib.npz",
                         camera_matrix=camera_matrix,
                         dist_coeffs=dist_coeffs)

                print("Saved calibration")
            else:
                print("Nothing to save")

        # =========================
        # R → reset frames
        # =========================
        elif key == ord('r'):
            objpoints = []
            imgpoints = []
            frame_count = 0
            print("Frames reset")

        # =========================
        # ESC → exit
        # =========================
        elif key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
