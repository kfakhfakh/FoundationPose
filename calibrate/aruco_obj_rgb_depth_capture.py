import argparse
import glob
import os
import time

import cv2
import numpy as np
try:
    import pyrealsense2 as rs
except Exception:
    rs = None


STEP_T = 0.001
STEP_R = 2.0
SAVE_WIDTH = 960
SAVE_HEIGHT = 540


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


def build_transform_from_euler_and_translation(rx_deg, ry_deg, rz_deg, tx, ty, tz, order='xyz'):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    if order == 'xyz':
        R = Rz @ Ry @ Rx
    elif order == 'zyx':
        R = Rx @ Ry @ Rz
    else:
        R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=float)
    return T


def project_points(K, T_cam_from_obj, pts_obj, dist_coeffs=None):
    rvec, _ = cv2.Rodrigues(T_cam_from_obj[:3, :3])
    tvec = T_cam_from_obj[:3, 3].reshape(3, 1)
    pts_obj = np.asarray(pts_obj, dtype=float).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(pts_obj, rvec, tvec, K, dist_coeffs)
    return imgpts.reshape(-1, 2).astype(int)


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
    cv2.line(vis, origin, tuple(pts[1]), (0, 0, 255), 3)
    cv2.line(vis, origin, tuple(pts[2]), (0, 255, 0), 3)
    cv2.line(vis, origin, tuple(pts[3]), (255, 0, 0), 3)


def marker_area(corner):
    pts = np.asarray(corner, dtype=np.float32).reshape(-1, 1, 2)
    return abs(cv2.contourArea(pts))


def adjust_transform_from_key(key, tx, ty, tz, rx, ry, rz):
    if key == ord('a'):
        tx -= STEP_T
    elif key == ord('z'):
        tx += STEP_T
    elif key == ord('e'):
        ty += STEP_T
    elif key == ord('r'):
        ty -= STEP_T
    elif key == ord('q'):
        tz += STEP_T
    elif key == ord('s'):
        tz -= STEP_T
    elif key == ord('d'):
        rx += STEP_R
    elif key == ord('f'):
        rx -= STEP_R
    elif key == ord('w'):
        ry -= STEP_R
    elif key == ord('x'):
        ry += STEP_R
    elif key == ord('c'):
        rz += STEP_R
    elif key == ord('v'):
        rz -= STEP_R
    return tx, ty, tz, rx, ry, rz


def get_next_capture_index(root_dir):
    rgb_dir = os.path.join(root_dir, 'rgb')
    max_index = -1
    if os.path.isdir(rgb_dir):
        for file_path in glob.glob(os.path.join(rgb_dir, '*')):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if base_name.isdigit():
                max_index = max(max_index, int(base_name))
    return max_index + 1


def ensure_output_dirs(output_root):
    rgb_dir = os.path.join(output_root, 'rgb')
    depth_dir = os.path.join(output_root, 'depth')
    cam_in_ob_dir = os.path.join(output_root, 'cam_in_ob')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(cam_in_ob_dir, exist_ok=True)
    return rgb_dir, depth_dir, cam_in_ob_dir


def save_capture(output_root, capture_index, rgb_frame, depth_frame, cam_in_ob):
    rgb_dir, depth_dir, cam_in_ob_dir = ensure_output_dirs(output_root)
    file_name = f'{capture_index:07d}'

    rgb_save = cv2.resize(rgb_frame, (SAVE_WIDTH, SAVE_HEIGHT), interpolation=cv2.INTER_AREA)
    depth_save = cv2.resize(depth_frame, (SAVE_WIDTH, SAVE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    depth_save = np.ascontiguousarray(depth_save)

    rgb_path = os.path.join(rgb_dir, f'{file_name}.png')
    depth_path = os.path.join(depth_dir, f'{file_name}.png')
    matrix_path = os.path.join(cam_in_ob_dir, f'{file_name}.txt')

    cv2.imwrite(rgb_path, rgb_save)
    cv2.imwrite(depth_path, depth_save)
    np.savetxt(matrix_path, cam_in_ob.reshape(4, 4), fmt='%.18e')

    return rgb_path, depth_path, matrix_path


def main():
    parser = argparse.ArgumentParser(description='Capture aligned RGB, aligned depth, and camera-in-object pose from ArUco marker (RealSense only)')
    parser.add_argument('--cam', type=int, default=0, help='(ignored) Camera device index — RealSense required')
    parser.add_argument('--marker-length', type=float, default=0.10, help='Marker side length in meters')
    parser.add_argument('--cam-k-file', type=str, default='C:\\Users\\kfakh\\OneDrive\\Desktop\\falku\\camera_calibration.npz', help='Optional camera intrinsics file (3x3 plain text)')
    parser.add_argument('--calib-file', type=str, default=None, help='Optional npz calibration file with camera_matrix and distortion_coefficients')
    parser.add_argument('--output-root', type=str, default=None, help='Output root directory. Default is ref_views_1 next to this script')
    parser.add_argument('--tx', type=float, default=0.0, help='Object offset X in marker frame (meters)')
    parser.add_argument('--ty', type=float, default=-0.10, help='Object offset Y in marker frame (meters)')
    parser.add_argument('--tz', type=float, default=0.015, help='Object offset Z in marker frame (meters)')
    parser.add_argument('--rx', type=float, default=0.0, help='Object rotation about X (deg) relative to marker')
    parser.add_argument('--ry', type=float, default=0.0, help='Object rotation about Y (deg) relative to marker')
    parser.add_argument('--rz', type=float, default=0.0, help='Object rotation about Z (deg) relative to marker')
    parser.add_argument('--axis-length', type=float, default=0.05, help='Length of axes to draw (meters)')
    parser.add_argument('--resize', type=str, default='1280x720', help='Resize output view as WIDTHxHEIGHT (e.g. 1280x720)')
    args = parser.parse_args()

    if rs is None:
        raise RuntimeError('pyrealsense2 is not installed or could not be imported — RealSense is required')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = args.output_root or os.path.join(script_dir, 'ref_views_1')
    os.makedirs(output_root, exist_ok=True)
    rgb_dir, depth_dir, cam_in_ob_dir = ensure_output_dirs(output_root)
    capture_index = get_next_capture_index(output_root)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        pipeline.stop()
        raise RuntimeError('Unable to get initial frames from RealSense')

    frame = np.asanyarray(color_frame.get_data())
    H, W = frame.shape[:2]

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
            try:
                loaded = np.load(args.cam_k_file, allow_pickle=True)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    if 'camera_matrix' in loaded:
                        K = loaded['camera_matrix']
                    elif 'arr_0' in loaded:
                        K = loaded['arr_0']
                    else:
                        keys = list(loaded.files)
                        K = loaded[keys[0]]
                else:
                    K = loaded
                K = np.asarray(K).reshape(3, 3)
                print(f'Loaded camera intrinsics from {args.cam_k_file} (numpy)')
            except Exception:
                K = np.loadtxt(args.cam_k_file, encoding='utf-8').reshape(3, 3)
                print(f'Loaded camera intrinsics from {args.cam_k_file} (text)')
        else:
            f = 0.9 * max(W, H)
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
            print('Using approximate intrinsics. Provide --cam-k-file or --calib-file for accurate results.')

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 20
    parameters.cornerRefinementMaxIterations = 100
    parameters.cornerRefinementMinAccuracy = 0.001
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    axis_len = args.axis_length
    tx = args.tx
    ty = args.ty
    tz = args.tz
    rx = args.rx
    ry = args.ry
    rz = args.rz

    print(f'Output root: {output_root}')
    print('Press Space to save RGB, depth, and camera-in-object matrix. Escape to quit.')

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        vis = rgb.copy()
        best_pose = None
        cam_in_obj = None

        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                try:
                    area = marker_area(corners[i])
                    rvec, tvec = estimate_marker_pose_from_corners(corners[i], args.marker_length, K, dist)
                    marker_in_cam = build_transform_from_rvec_tvec(rvec, tvec)
                    cam_in_marker = np.linalg.inv(marker_in_cam)
                    draw_axis(vis, K, rvec, tvec, axis_len, dist)

                    T_marker_from_obj = build_transform_from_euler_and_translation(rx, ry, rz, tx, ty, tz)
                    ob_in_cam = marker_in_cam @ T_marker_from_obj
                    cam_in_obj = np.linalg.inv(ob_in_cam)

                    cam_pos_in_obj = cam_in_obj[:3, 3]
                    cam_rot_in_obj = cam_in_obj[:3, :3]
                    cam_euler_in_obj = rotation_matrix_to_euler(cam_rot_in_obj)

                    if best_pose is None or area > best_pose['area']:
                        best_pose = {
                            'id': int(marker_id),
                            'area': area,
                            'cam_in_marker': cam_in_marker,
                            'cam_pos_in_marker': cam_in_marker[:3, 3],
                            'ob_in_cam': ob_in_cam,
                            'cam_in_obj': cam_in_obj,
                            'cam_pos_in_obj': cam_pos_in_obj,
                            'cam_euler_in_obj': cam_euler_in_obj,
                        }

                    pts_obj = np.array([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
                    imgpts = project_points(K, ob_in_cam, pts_obj, dist)
                    origin = tuple(imgpts[0])
                    xpt = tuple(imgpts[1])
                    ypt = tuple(imgpts[2])
                    zpt = tuple(imgpts[3])

                    cv2.line(vis, origin, xpt, (0, 0, 255), 3)
                    cv2.line(vis, origin, ypt, (0, 255, 0), 3)
                    cv2.line(vis, origin, zpt, (255, 0, 0), 3)

                    txt = f'id:{int(marker_id)} tx:{tx:.3f} ty:{ty:.3f} tz:{tz:.3f} rx:{rx:.1f} ry:{ry:.1f} rz:{rz:.1f}'
                    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cam_pos_txt = f'Cam Pos (m): x={cam_pos_in_obj[0]:.4f} y={cam_pos_in_obj[1]:.4f} z={cam_pos_in_obj[2]:.4f}'
                    cv2.putText(vis, cam_pos_txt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    cam_rot_txt = f'Cam Rot (deg): rx={cam_euler_in_obj[0]:.1f} ry={cam_euler_in_obj[1]:.1f} rz={cam_euler_in_obj[2]:.1f}'
                    cv2.putText(vis, cam_rot_txt, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    cv2.putText(vis, 'Space save  Escape quit', (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f'Error processing marker {i}: {e}')
        else:
            cv2.putText(vis, 'No marker detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis, 'Space save  Escape quit', (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if best_pose is not None:
            cv2.putText(vis, f'Using marker {best_pose["id"]} (best area)', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset = 115
            cam_marker_txt = f'Cam in Marker (m): x={best_pose["cam_pos_in_marker"][0]:.4f} y={best_pose["cam_pos_in_marker"][1]:.4f} z={best_pose["cam_pos_in_marker"][2]:.4f}'
            cv2.putText(vis, cam_marker_txt, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        try:
            rw, rh = args.resize.split('x')
            rw, rh = int(rw), int(rh)
        except Exception:
            rw, rh = vis.shape[1], vis.shape[0]

        vis_resized = cv2.resize(vis, (rw, rh))
        cv2.imshow('ArUco RGB-D Capture', vis_resized)
        k = cv2.waitKey(1) & 0xFF

        if k == 0x1b:
            break

        if k == ord(' '):
            if best_pose is None or cam_in_obj is None:
                print('No marker detected in current frame')
            else:
                rgb_path, depth_path, matrix_path = save_capture(output_root, capture_index, rgb, depth, cam_in_obj)
                print(f'Saved capture {capture_index:07d}')
                print(f'  RGB: {rgb_path}')
                print(f'  Depth: {depth_path}')
                print(f'  Matrix: {matrix_path}')
                capture_index += 1

        tx, ty, tz, rx, ry, rz = adjust_transform_from_key(k, tx, ty, tz, rx, ry, rz)

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
