import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
IMAGE_FOLDER = "C:\\Users\\kfakh\\OneDrive\\Desktop\\falku\\calibration_images\\*.png"
CALIB_FILE = "camera_calibration.npz"
CHESSBOARD_SIZE = (8, 5)

images = glob.glob(IMAGE_FOLDER)

# =========================
# LOAD CALIBRATION
# =========================
data = np.load(CALIB_FILE)
K = data["camera_matrix"]
dist = data["distortion_coefficients"]

# =========================
# DRAW LINES ON CORNERS
# =========================
def draw_lines(img, corners):
    corners = corners.reshape(CHESSBOARD_SIZE[1], CHESSBOARD_SIZE[0], 2)
    img = img.copy()

    for row in range(CHESSBOARD_SIZE[1]):
        for col in range(CHESSBOARD_SIZE[0] - 1):
            pt1 = tuple(corners[row][col].astype(int))
            pt2 = tuple(corners[row][col + 1].astype(int))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    return img

# =========================
# PROCESS IMAGES
# =========================
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]

    # UNDISTORT FIRST
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undist = cv2.undistort(img, K, dist, None, new_K)

    gray_u = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # FIND CORNERS AGAIN (IMPORTANT FIX)
    ret, corners_u = cv2.findChessboardCorners(gray_u, CHESSBOARD_SIZE, None)

    if not ret:
        print(f"Chessboard not found in undistorted image: {fname}")
        continue

    corners_u = cv2.cornerSubPix(
        gray_u, corners_u, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # ORIGINAL corners for comparison
    ret, corners_o = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    corners_o = cv2.cornerSubPix(
        gray, corners_o, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # DRAW LINES
    orig_vis = draw_lines(img, corners_o)
    undist_vis = draw_lines(undist, corners_u)

    # resize for viewing
    orig_vis = cv2.resize(orig_vis, (800, 600))
    undist_vis = cv2.resize(undist_vis, (800, 600))

    # convert for matplotlib
    orig_vis = cv2.cvtColor(orig_vis, cv2.COLOR_BGR2RGB)
    undist_vis = cv2.cvtColor(undist_vis, cv2.COLOR_BGR2RGB)

    # =========================
    # DISPLAY (ZOOMABLE)
    # =========================
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].imshow(orig_vis)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(undist_vis)
    ax[1].set_title("Undistorted")
    ax[1].axis("off")

    plt.suptitle(fname)
    plt.tight_layout()
    plt.show()
