import argparse
import glob
import os

import cv2
import numpy as np


def load_k_matrix(k_path):
    k = np.loadtxt(k_path, dtype=np.float64)
    return np.asarray(k, dtype=np.float64).reshape(3, 3)


def load_cam_in_ob(matrix_path):
    mat = np.loadtxt(matrix_path, dtype=np.float64)
    return np.asarray(mat, dtype=np.float64).reshape(4, 4)


def draw_axes(image, K, cam_in_ob, axis_len=0.05, dist=None, debug=False):
    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)

    # cam_in_ob is the inverse of ob_in_cam (camera pose in object frame).
    # To project object points, we need: ob_in_cam = inv(cam_in_ob)
    # For SE(3): inv([R, t]) = [R^T, -R^T @ t]
    R_cam_in_ob = cam_in_ob[:3, :3]
    t_cam_in_ob = cam_in_ob[:3, 3].reshape(3, 1)
    
    # Compute ob_in_cam for projection
    R_ob_in_cam = R_cam_in_ob.T
    t_ob_in_cam = -R_ob_in_cam @ t_cam_in_ob
    
    rvec, _ = cv2.Rodrigues(R_ob_in_cam)
    tvec = t_ob_in_cam

    if debug:
        print(f"cam_in_ob tvec:\n{t_cam_in_ob.flatten()}")
        print(f"tvec used for projection (computed as -R^T @ t):\n{tvec.flatten()}")

    axis_points = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ], dtype=np.float64)

    img_pts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)
    pts = img_pts.reshape(-1, 2).astype(int)

    if debug:
        print(f"Projected origin: {pts[0]}")
        print(f"Image shape: {image.shape}")

    origin = tuple(pts[0])
    x_axis = tuple(pts[1])
    y_axis = tuple(pts[2])
    z_axis = tuple(pts[3])

    out = image.copy()
    cv2.line(out, origin, x_axis, (0, 0, 255), 3)
    cv2.line(out, origin, y_axis, (0, 255, 0), 3)
    cv2.line(out, origin, z_axis, (255, 0, 0), 3)
    cv2.circle(out, origin, 5, (255, 255, 255), -1)
    return out


def find_matching_files(root_dir):
    rgb_dir = os.path.join(root_dir, "rgb")
    cam_dir = os.path.join(root_dir, "cam_in_ob")
    k_path = os.path.join(root_dir, "K.txt")

    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"Missing rgb directory: {rgb_dir}")
    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"Missing cam_in_ob directory: {cam_dir}")
    if not os.path.isfile(k_path):
        raise FileNotFoundError(f"Missing K.txt file: {k_path}")

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.*")))
    cam_map = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob.glob(os.path.join(cam_dir, "*.txt"))
    }

    return rgb_files, cam_map, k_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project object axes onto RGB images using cam_in_ob and K.txt from a dataset folder"
    )
    parser.add_argument(
        "--root",
        default="C:\\Users\\kfakh\\OneDrive\\Desktop\\falku\\FoundationPose\\video_utils\\ref_views_16\\ob_0000001",
        help="Root folder containing rgb/, cam_in_ob/, and K.txt"
    )
    parser.add_argument(
        "--axis-len",
        type=float,
        default=0.05,
        help="Axis length in object units, usually meters"
    )
    parser.add_argument(
        "--output",
        default="C:\\Users\\kfakh\\OneDrive\\Desktop\\falku\\FoundationPose\\video_utils\\ref_views_16\\ob_0000001\\out",
        help="Output folder for annotated images. Default: <root>/axis_overlay"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show images while processing"
    )
    parser.add_argument(
        "--dist",
        default=None,
        help="Optional 5-value distortion coefficients file or .npz containing distortion_coefficients"
    )
    parser.add_argument(
        "--k-resolution",
        default=None,
        help="Resolution K was computed for (WIDTHxHEIGHT, e.g. 1920x1080). If provided and differs from image size, K will be scaled."
    )
    return parser.parse_args()


def load_distortion(dist_path):
    if dist_path is None:
        return np.zeros((5, 1), dtype=np.float64)

    if not os.path.isfile(dist_path):
        raise FileNotFoundError(f"Missing distortion file: {dist_path}")

    if dist_path.lower().endswith(".npz"):
        data = np.load(dist_path, allow_pickle=True)
        if "distortion_coefficients" in data:
            dist = data["distortion_coefficients"]
        elif "dist" in data:
            dist = data["dist"]
        else:
            first_key = list(data.files)[0]
            dist = data[first_key]
    else:
        dist = np.loadtxt(dist_path, dtype=np.float64)

    dist = np.asarray(dist, dtype=np.float64)
    return dist.reshape(-1, 1)


def scale_K_for_image(K, k_resolution, image_shape):
    """Scale intrinsic matrix K from k_resolution to image_shape."""
    if k_resolution is None:
        # Default: assume K was computed for Full HD (1920x1080)
        k_w, k_h = 1920, 1080
        print(f"Assuming K was computed for Full HD: {k_w}x{k_h}")
    else:
        try:
            k_w, k_h = map(int, k_resolution.split('x'))
        except Exception:
            print(f"Invalid --k-resolution format: {k_resolution}. Should be WIDTHxHEIGHT")
            return K
    
    img_h, img_w = image_shape[:2]
    scale_x = img_w / k_w
    scale_y = img_h / k_h
    
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    
    print(f"Scaled K from {k_w}x{k_h} to {img_w}x{img_h} (scale_x={scale_x:.3f}, scale_y={scale_y:.3f})")
    print(f"Scaled K matrix:\n{K_scaled}")
    return K_scaled


def main():
    args = parse_args()
    root = args.root
    output_dir = args.output or os.path.join(root, "axis_overlay")
    os.makedirs(output_dir, exist_ok=True)

    rgb_files, cam_map, k_path = find_matching_files(root)
    K = load_k_matrix(k_path)
    dist = load_distortion(args.dist)

    count = 0
    for rgb_path in rgb_files:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        cam_path = cam_map.get(stem)
        if cam_path is None:
            print(f"Skipping {stem}: no matching cam_in_ob file")
            continue

        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping {rgb_path}: could not read image")
            continue

        try:
            cam_in_ob = load_cam_in_ob(cam_path)
        except Exception as exc:
            print(f"Skipping {stem}: failed to load matrix: {exc}")
            continue

        K_for_image = scale_K_for_image(K, args.k_resolution, image.shape)
        annotated = draw_axes(image, K_for_image, cam_in_ob, axis_len=args.axis_len, dist=dist, debug=(count == 0))
        out_path = os.path.join(output_dir, os.path.basename(rgb_path))
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")
        count += 1

        if args.show:
            cv2.imshow("Axis Overlay", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()
    print(f"Done. Processed {count} image(s).")


if __name__ == "__main__":
    main()
