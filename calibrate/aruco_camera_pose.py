import argparse
import os
import numpy as np
import cv2


def estimate_marker_pose_from_corners(corner, marker_length, K, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=float)

    half = marker_length / 2.0
    obj_pts = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=float)

    img_pts = np.asarray(corner, dtype=float).reshape(4, 2)
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success:
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError('Failed to estimate marker pose')
    return rvec.reshape(3, 1), tvec.reshape(3, 1)


def build_transform_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def rotation_matrix_to_euler(R):
    """Extract Euler angles (roll, pitch, yaw) in degrees from rotation matrix."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])


def draw_axis(vis, K, rvec, tvec, axis_len, dist_coeffs=None):
    axis_pts = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ], dtype=float)
    imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist_coeffs)
    pts = imgpts.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    cv2.line(vis, origin, tuple(pts[1]), (0, 0, 255), 3)  # X - red
    cv2.line(vis, origin, tuple(pts[2]), (0, 255, 0), 3)  # Y - green
    cv2.line(vis, origin, tuple(pts[3]), (255, 0, 0), 3)  # Z - blue


def main():
    parser = argparse.ArgumentParser(description='Detect ArUco marker and get camera pose relative to marker')
    parser.add_argument('--cam', type=int, default=0, help='Camera device index')
    parser.add_argument('--marker-length', type=float, default=0.10, help='Marker side length in meters')
    parser.add_argument('--cam-k-file', type=str, default=None, help='Optional camera intrinsics file (3x3 plain text)')
    parser.add_argument('--calib-file', type=str, default=None, help='Optional npz calibration file with camera_matrix and distortion_coefficients')
    parser.add_argument('--axis-length', type=float, default=0.05, help='Length of axes to draw (meters)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError('Unable to open camera')

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError('Unable to read from camera')
    H, W = frame.shape[:2]

    # Load calibration if available (npz saved by calibrate script) or fall back to cam_k_file or approximate
    dist = None
    K = None
    if args.calib_file is not None and os.path.exists(args.calib_file):
        data = np.load(args.calib_file)
        if 'camera_matrix' in data and 'distortion_coefficients' in data:
            K = data['camera_matrix']
            dist = data['distortion_coefficients']
            print(f'Loaded calibration from {args.calib_file}')
        else:
            print(f'Calibration file {args.calib_file} does not contain expected keys. Falling back.')

    if K is None:
        if args.cam_k_file is not None and os.path.exists(args.cam_k_file):
            K = np.loadtxt(args.cam_k_file).reshape(3, 3)
            print(f'Loaded camera intrinsics from {args.cam_k_file}')
        else:
            f = 0.9 * max(W, H)
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
            print('Using approximate intrinsics. Provide --cam-k-file or --calib-file for accurate results.')

    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    axis_len = args.axis_length

    print('Camera pose relative to ArUco marker (in cm and degrees)')
    print('Press space to print current pose. q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        vis = frame.copy()

        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                try:
                    rvec, tvec = estimate_marker_pose_from_corners(corners[i], args.marker_length, K, dist)
                    
                    # marker_in_cam: where marker is relative to camera
                    marker_in_cam = build_transform_from_rvec_tvec(rvec, tvec)
                    
                    # cam_in_marker: where camera is relative to marker (inverse)
                    cam_in_marker = np.linalg.inv(marker_in_cam)
                    
                    # Draw marker axis
                    draw_axis(vis, K, rvec, tvec, axis_len, dist)
                    
                    # Extract camera translation (in meters, convert to cm) and rotation
                    cam_pos_m = cam_in_marker[:3, 3]
                    cam_pos_cm = cam_pos_m * 100  # convert to cm
                    cam_rot_matrix = cam_in_marker[:3, :3]
                    cam_euler = rotation_matrix_to_euler(cam_rot_matrix)
                    
                    # Display on image
                    txt = f'Marker {int(marker_id)}'
                    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    txt_pos = f'Cam Pos (cm): X={cam_pos_cm[0]:7.2f}  Y={cam_pos_cm[1]:7.2f}  Z={cam_pos_cm[2]:7.2f}'
                    cv2.putText(vis, txt_pos, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    txt_rot = f'Cam Rot (deg): Roll={cam_euler[0]:7.2f}  Pitch={cam_euler[1]:7.2f}  Yaw={cam_euler[2]:7.2f}'
                    cv2.putText(vis, txt_rot, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    help_txt = 'Space to print  q to quit'
                    cv2.putText(vis, help_txt, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                except Exception as e:
                    print(f'Error processing marker {i}: {e}')
        else:
            cv2.putText(vis, 'No marker detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('ArUco Camera Pose', vis)
        k = cv2.waitKey(1) & 0xFF
        
        if k == ord('q'):
            break
        
        if k == ord(' '):  # space key
            if ids is not None and len(ids) > 0:
                try:
                    rvec, tvec = estimate_marker_pose_from_corners(corners[0], args.marker_length, K, dist)
                    marker_in_cam = build_transform_from_rvec_tvec(rvec, tvec)
                    cam_in_marker = np.linalg.inv(marker_in_cam)
                    
                    cam_pos_m = cam_in_marker[:3, 3]
                    cam_pos_cm = cam_pos_m * 100
                    cam_rot_matrix = cam_in_marker[:3, :3]
                    cam_euler = rotation_matrix_to_euler(cam_rot_matrix)
                    
                    print('\n' + '='*70)
                    print('Camera Pose relative to ArUco Marker')
                    print('='*70)
                    print(f'Translation (cm):')
                    print(f'  X: {cam_pos_cm[0]:10.4f} cm')
                    print(f'  Y: {cam_pos_cm[1]:10.4f} cm')
                    print(f'  Z: {cam_pos_cm[2]:10.4f} cm')
                    print(f'Distance from marker: {np.linalg.norm(cam_pos_cm):10.4f} cm')
                    print(f'\nRotation (Euler angles in degrees):')
                    print(f'  Roll  (X): {cam_euler[0]:10.4f}°')
                    print(f'  Pitch (Y): {cam_euler[1]:10.4f}°')
                    print(f'  Yaw   (Z): {cam_euler[2]:10.4f}°')
                    print(f'\nFull 4x4 camera-in-marker matrix:')
                    np.set_printoptions(precision=6, suppress=True)
                    print(cam_in_marker)
                    print('='*70 + '\n')
                except Exception as e:
                    print(f'Error: {e}')
            else:
                print('No marker detected in current frame')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
