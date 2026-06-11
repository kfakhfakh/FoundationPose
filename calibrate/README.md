# Calibration Utilities

This folder contains scripts for RealSense camera calibration, ArUco-based pose estimation, and calibration quality checks.

## Files Overview

- `realsense_calib_images_capture.py`
- `calibrate_cam.py`
- `calibrate_cam_folder.py`
- `calib_visual_check.py`
- `realsense_intrensec.py`
- `aruco_camera_pose.py`
- `aruco_obj_pose.py`
- `camera_calibration.npz` (generated calibration output)

---

## 1) Capture calibration images

### `realsense_calib_images_capture.py`
Captures RGB images from a RealSense camera for chessboard calibration.

- Uses checkerboard settings:
  - `CHECKERBOARD = (8, 5)` (inner corners)
- Press:
  - `c` to save current frame into `calibration_images/`
  - `ESC` to quit

Run:
```bash
python realsense_calib_images_capture.py
```

Note: Paths and settings are script-level constants.

---

## 2) Camera calibration from captured images

### `calibrate_cam.py`
RealSense live calibration utility (interactive).

Controls:
- `SPACE`: capture detected chessboard frame
- `c`: calibrate using captured frames
- `l`: load existing `realsense_calib.npz`
- `e`: evaluate loaded calibration (reprojection error)
- `s`: save calibration to `realsense_calib.npz`
- `r`: reset captured frames
- `ESC`: quit

Run:
```bash
python calibrate_cam.py
```

### `calibrate_cam_folder.py`
Runs OpenCV camera calibration from a folder of chessboard images and saves calibration.

- Inputs are configured at the top of the script:
  - `IMAGE_FOLDER` (glob path)
  - `CHESSBOARD_SIZE`
  - `SQUARE_SIZE`
- Shows side-by-side original vs undistorted images.
- Prints per-frame and overall reprojection error.
- Saves output to `.npz` with:
  - `camera_matrix`
  - `distortion_coefficients`
  - `rvecs`, `tvecs`
  - `reprojection_error`

Run:
```bash
python calibrate_cam_folder.py
```

---

## 3) Calibration quality check

### `calib_visual_check.py`
Loads calibration and visualizes chessboard line straightness before/after undistortion.

- Uses matplotlib for side-by-side display of:
  - original image with chessboard lines
  - undistorted image with chessboard lines
- Useful for visually validating distortion correction.

Run:
```bash
python calib_visual_check.py
```

---

## 4) Print RealSense intrinsics

### `realsense_intrensec.py`
Prints RealSense color/depth intrinsics and outputs the color camera matrix `K`.

Run:
```bash
python realsense_intrensec.py
```

---

## 5) ArUco pose tools

### `aruco_camera_pose.py`
Detects ArUco markers and estimates **camera pose relative to marker** (`cam_in_marker`).

Features:
- RealSense-only stream
- Loads intrinsics from:
  - `--calib-file` (`.npz` with `camera_matrix`, `distortion_coefficients`) or
  - `--cam-k-file` (text or numpy format)
- Draws marker axes and overlays translation/rotation
- `SPACE` prints full pose details and 4x4 matrix
- `q` quits

Run example:
```bash
python aruco_camera_pose.py --marker-length 0.1 --calib-file camera_calibration.npz
```

### `aruco_obj_pose.py`
Detects ArUco marker and computes **camera pose in object frame** (`cam_in_obj`) using user-defined marker→object transform offsets.

Interactive transform tuning:
- Translation:
  - `a/z`: X -/+
  - `e/r`: Y +/−
  - `q/s`: Z +/−
- Rotation:
  - `d/f`: Rx +/−
  - `w/x`: Ry −/+
  - `c/v`: Rz +/−
- `SPACE`: capture current `cam_in_obj` matrix into `captured_poses/`
- `ESC`: quit

Run example:
```bash
python aruco_obj_pose.py --marker-length 0.1 --calib-file camera_calibration.npz
```

---

## Output files

- `camera_calibration.npz` (or `realsense_calib.npz` in some scripts)
- `captured_poses/cam_in_obj_*.txt` (from `aruco_obj_pose.py`)
- Optional saved visualization frames if `--save-dir` is provided in ArUco scripts

Compatibility note:
- `aruco_camera_pose.py` and `aruco_obj_pose.py` expect `--calib-file` to contain keys `camera_matrix` and `distortion_coefficients`.
- `calibrate_cam.py` saves `dist_coeffs` (different key name), so for those ArUco scripts either:
  - use `camera_calibration.npz` from `calibrate_cam_folder.py`, or
  - pass `--cam-k-file` and load only intrinsics matrix `K`.

---

## Notes

- Most scripts are tied to RealSense (`pyrealsense2`) and will fail without it.
- Several scripts use hardcoded paths/settings. Update constants before running.
- Checkerboard dimensions must match your printed board exactly (inner corners and square size units).

## Very important note about intrinsics and resolution

Camera intrinsics are resolution-dependent.

If you calibrate at one resolution (for example, 1920×1080), you must use the same resolution later when running pose estimation with that calibration.

If you change resolution after calibration, the original intrinsic matrix is no longer directly valid unless you properly scale/adapt it to the new image size.
