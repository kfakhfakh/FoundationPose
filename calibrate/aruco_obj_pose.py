import argparse
import os
import time
import numpy as np
import cv2


STEP_T = 0.001
STEP_R = 2.0


def build_transform_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def build_transform_from_euler_and_translation(rx_deg, ry_deg, rz_deg, tx, ty, tz, order='xyz'):
    # rotations in degrees applied in order x then y then z by default
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
    # pts_obj: (N,3)
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


def draw_axis(vis, K, rvec, tvec, axis_len):
    axis_pts = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ], dtype=float)
    imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, None)
    pts = imgpts.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    cv2.line(vis, origin, tuple(pts[1]), (0, 0, 255), 3)
    cv2.line(vis, origin, tuple(pts[2]), (0, 255, 0), 3)
    cv2.line(vis, origin, tuple(pts[3]), (255, 0, 0), 3)


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


def main():
    parser = argparse.ArgumentParser(description='Detect ArUco marker and draw object axis according to offset and rotations')
    parser.add_argument('--cam', type=int, default=0, help='Camera device index')
    parser.add_argument('--marker-length', type=float, default=0.10, help='Marker side length in meters')
    parser.add_argument('--cam-k-file', type=str, default=None, help='Optional camera intrinsics file (3x3 plain text)')
    parser.add_argument('--calib-file', type=str, default=None, help='Optional npz calibration file with camera_matrix and distortion_coefficients')
    parser.add_argument('--tx', type=float, default=0.0, help='Object offset X in marker frame (meters)')
    parser.add_argument('--ty', type=float, default=-0.1, help='Object offset Y in marker frame (meters)')
    parser.add_argument('--tz', type=float, default=0.015 , help='Object offset Z in marker frame (meters)')
    parser.add_argument('--rx', type=float, default=0.0, help='Object rotation about X (deg) relative to marker')
    parser.add_argument('--ry', type=float, default=0.0, help='Object rotation about Y (deg) relative to marker')
    parser.add_argument('--rz', type=float, default=0.0, help='Object rotation about Z (deg) relative to marker')
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
            # approximate focal length
            f = 0.9 * max(W, H)
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
            print('Using approximate intrinsics. Provide --cam-k-file or --calib-file for accurate results.')

    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    axis_len = args.axis_length
    tx = args.tx
    ty = args.ty
    tz = args.tz
    rx = args.rx
    ry = args.ry
    rz = args.rz

    out_dir = 'captured_poses'
    os.makedirs(out_dir, exist_ok=True)

    print('Press c to capture and print object-in-camera matrix. q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        vis = frame.copy()

        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec = estimate_marker_pose_from_corners(corners[i], args.marker_length, K)
                T_cam_from_marker = build_transform_from_rvec_tvec(rvec, tvec)

                # draw marker axis
                draw_axis(vis, K, rvec, tvec, axis_len)

                # build object transform relative to marker using user offsets/rotations
                T_marker_from_obj = build_transform_from_euler_and_translation(
                    rx, ry, rz, tx, ty, tz)

                # object in camera
                ob_in_cam = T_cam_from_marker @ T_marker_from_obj

                # draw object axes: origin and unit axes scaled by axis_len
                pts_obj = np.array([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
                imgpts = project_points(K, ob_in_cam, pts_obj)
                origin = tuple(imgpts[0])
                xpt = tuple(imgpts[1])
                ypt = tuple(imgpts[2])
                zpt = tuple(imgpts[3])

                cv2.line(vis, origin, xpt, (0, 0, 255), 3)  # X - red
                cv2.line(vis, origin, ypt, (0, 255, 0), 3)  # Y - green
                cv2.line(vis, origin, zpt, (255, 0, 0), 3)  # Z - blue

                # overlay text
                txt = f'id:{int(marker_id)} tx:{tx:.3f} ty:{ty:.3f} tz:{tz:.3f} rx:{rx:.1f} ry:{ry:.1f} rz:{rz:.1f}'
                cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                help_txt = 'a/z x  e/r y  q/s z  d/f rx  w/x ry  c/v rz  c capture  q quit'
                cv2.putText(vis, help_txt, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        else:
            cv2.putText(vis, 'No marker detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            help_txt = 'a/z x  e/r y  q/s z  d/f rx  w/x ry  c/v rz  c capture  q quit'
            cv2.putText(vis, help_txt, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('ArUco Object Pose', vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 0x27 :
            break
        if k == ord(' '):
            # capture: if marker present, save ob_in_cam matrix and print
            if ids is None or len(ids) == 0:
                print('No marker to capture in this frame')
            else:
                # use the first marker detected
                rvec, tvec = estimate_marker_pose_from_corners(corners[0], args.marker_length, K)
                T_cam_from_marker = build_transform_from_rvec_tvec(rvec, tvec)
                T_marker_from_obj = build_transform_from_euler_and_translation(
                    rx, ry, rz, tx, ty, tz)
                ob_in_cam = T_cam_from_marker @ T_marker_from_obj
                ts = int(time.time() * 1000)
                fname = os.path.join(out_dir, f'ob_in_cam_{ts}.txt')
                np.savetxt(fname, ob_in_cam.reshape(4, 4), fmt='%.18e')
                print('Saved', fname)
                print('ob_in_cam matrix:')
                np.set_printoptions(precision=6, suppress=True)
                print(ob_in_cam)

        tx, ty, tz, rx, ry, rz = adjust_transform_from_key(k, tx, ty, tz, rx, ry, rz)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
