import argparse
import os
import numpy as np
import cv2
import time
try:
    import pyrealsense2 as rs
except Exception:
    rs = None


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


def marker_area(corner):
    pts = np.asarray(corner, dtype=np.float32).reshape(-1, 1, 2)
    return abs(cv2.contourArea(pts))


def main():
    parser = argparse.ArgumentParser(description='Detect ArUco marker and get camera pose relative to marker (RealSense only)')
    parser.add_argument('--cam', type=int, default=0, help='(ignored) Camera device index — RealSense required')
    parser.add_argument('--marker-length', type=float, default=0.1, help='Marker side length in meters')
    parser.add_argument('--cam-k-file', type=str, default="C:\\Users\\kfakh\\OneDrive\\Desktop\\falku\\camera_calibration.npz", help='Optional camera intrinsics file (3x3 plain text)')
    parser.add_argument('--calib-file', type=str, default=None, help='Optional npz calibration file with camera_matrix and distortion_coefficients')
    parser.add_argument('--axis-length', type=float, default=0.05, help='Length of axes to draw (meters)')
    parser.add_argument('--resize', type=str, default='1280x720', help='Resize output view as WIDTHxHEIGHT (e.g. 1280x720)')
    parser.add_argument('--save-dir', type=str, default=None, help='If set, save annotated resized frames to this directory')
    args = parser.parse_args()

    # RealSense-only initialization
    if rs is None:
        raise RuntimeError('pyrealsense2 is not installed or could not be imported — RealSense is required')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    # prime one frame
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    if not color_frame:
        pipeline.stop()
        raise RuntimeError('Unable to get color frame from RealSense')
    frame = np.asanyarray(color_frame.get_data())
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
            # Try loading as numpy file (.npz/.npy) first, then fallback to text
            try:
                loaded = np.load(args.cam_k_file, allow_pickle=True)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    if 'camera_matrix' in loaded:
                        K = loaded['camera_matrix']
                    elif 'arr_0' in loaded:
                        K = loaded['arr_0']
                    else:
                        # pick the first array available
                        keys = list(loaded.files)
                        K = loaded[keys[0]]
                else:
                    K = loaded
                K = np.asarray(K).reshape(3, 3)
                print(f'Loaded camera intrinsics from {args.cam_k_file} (numpy)')
            except Exception:
                # fallback to text loading (specify utf-8)
                K = np.loadtxt(args.cam_k_file, encoding='utf-8').reshape(3, 3)
                print(f'Loaded camera intrinsics from {args.cam_k_file} (text)')
        else:
            f = 0.9 * max(W, H)
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
            print('Using approximate intrinsics. Provide --cam-k-file or --calib-file for accurate results.')

    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 20
    parameters.cornerRefinementMaxIterations = 100
    parameters.cornerRefinementMinAccuracy = 0.001
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    axis_len = args.axis_length

    print('Camera pose relative to ArUco marker (in cm and degrees)')
    print('Press space to print current pose. q to quit.')

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        vis = frame.copy()

        best_pose = None
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                try:
                    area = marker_area(corners[i])
                    rvec, tvec = estimate_marker_pose_from_corners(corners[i], args.marker_length, K, dist)
                    
                    # marker_in_cam: where marker is relative to camera
                    marker_in_cam = build_transform_from_rvec_tvec(rvec, tvec)
                    
                    # cam_in_marker: where camera is relative to marker (inverse)
                    cam_in_marker = np.linalg.inv(marker_in_cam)
                    
                    # Draw marker axis
                    draw_axis(vis, K, rvec, tvec, axis_len, dist)

                    if best_pose is None or area > best_pose['area']:
                        best_pose = {
                            'id': int(marker_id),
                            'area': area,
                            'rvec': rvec,
                            'tvec': tvec,
                            'cam_in_marker': cam_in_marker,
                        }
                    
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

        if best_pose is not None:
            cv2.putText(vis, f'Using marker {best_pose["id"]} (best area)', (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Resize, optionally save, then visualize
        try:
            rw, rh = args.resize.split('x')
            rw, rh = int(rw), int(rh)
        except Exception:
            rw, rh = vis.shape[1], vis.shape[0]

        vis_resized = cv2.resize(vis, (rw, rh))

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            fname = os.path.join(args.save_dir, f'aruco_{ts}.png')
            cv2.imwrite(fname, vis_resized)

        cv2.imshow('ArUco Camera Pose', vis_resized)
        k = cv2.waitKey(1) & 0xFF
        
        if k == ord('q'):
            break
        
        if k == ord(' '):  # space key
            if best_pose is not None:
                try:
                    cam_in_marker = best_pose['cam_in_marker']
                    
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
                    print(f'  Marker used: {best_pose["id"]}  area={best_pose["area"]:.1f} px^2')
                    print(f'\nFull 4x4 camera-in-marker matrix:')
                    np.set_printoptions(precision=6, suppress=True)
                    print(cam_in_marker)
                    print('='*70 + '\n')
                except Exception as e:
                    print(f'Error: {e}')
            else:
                print('No marker detected in current frame')

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
