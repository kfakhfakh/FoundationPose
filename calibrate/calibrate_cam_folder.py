import cv2
import numpy as np
import glob

# =========================
# USER SETTINGS
# =========================
CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE = 3.0
IMAGE_FOLDER = "C:\\Users\\kfakh\\OneDrive\\Desktop\\falku\\calibration_images\\*.png"
OUTPUT_FILE = "camera_calibration.npz"

# =========================
# OBJECT POINTS
# =========================
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob(IMAGE_FOLDER)

if len(images) == 0:
    print("No images found!")
    exit()

gray_shape = None

# =========================
# DETECT CORNERS 
# =========================
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray_shape is None:
        gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        imgpoints.append(corners2)

# =========================
# CALIBRATION
# =========================
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None
)

# =========================
# PER-FRAME + TOTAL ERROR
# =========================
total_error = 0
total_points = 0

# =========================
# UNDISTORT + COMPARE
# =========================
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    undistorted = cv2.undistort(img, K, dist, None, new_K)

    # Reprojection error for this frame
    projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    
    # Calculate per-point errors (pixels)
    error_per_point = np.linalg.norm(imgpoints[i] - projected, axis=2).flatten()
    mean_error = np.mean(error_per_point)
    
    # Accumulate for overall calculation
    total_error += np.sum(error_per_point)
    total_points += len(error_per_point)

    print(f"Frame {i+1}/{len(images)} - Reprojection Error: {mean_error:.6f}")

    # =========================
    # SIDE-BY-SIDE DISPLAY
    # =========================
    combined = np.hstack((
        cv2.resize(img, (800, 600)),
        cv2.resize(undistorted, (800, 600))
    ))

    cv2.putText(combined, "Original", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(combined, "Undistorted", (850, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Original vs Undistorted", combined)

    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

# =========================
# FINAL ERROR
# =========================
overall_error = total_error / total_points

print("\n===== FINAL RESULTS =====")
print("Camera Matrix:\n", K)
print("\nDistortion Coefficients:\n", dist.ravel())
print("\nOverall Reprojection Error:", overall_error)

# =========================
# SAVE
# =========================
np.savez(
    OUTPUT_FILE,
    camera_matrix=K,
    distortion_coefficients=dist,
    rvecs=rvecs,
    tvecs=tvecs,
    reprojection_error=overall_error
)

print("\nSaved to:", OUTPUT_FILE)
