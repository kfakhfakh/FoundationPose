import os
import argparse
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def apply_mask_to_img(img, mask):
    if mask is None:
        return img

    # ensure mask size
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if img.ndim == 2:
        return img * mask
    elif img.ndim == 3:
        channels = img.shape[2]
        mask_expanded = np.repeat(mask[:, :, None], channels, axis=2)
        return img * mask_expanded
    else:
        raise RuntimeError(f"Unsupported image shape: {img.shape}")


def run_dir_mode(input_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in sorted(os.listdir(input_dir)):
        if not file.lower().endswith((".png", ".jpg", ".tif", ".tiff")):
            continue

        img_path = os.path.join(input_dir, file)
        mask_path = os.path.join(mask_dir, os.path.splitext(file)[0] + ".png")

        if not os.path.exists(mask_path):
            print(f"Missing mask for {file}")
            continue

        img = load_image(img_path)
        mask = load_mask(mask_path)
        if img is None or mask is None:
            continue

        masked = apply_mask_to_img(img, mask)
        out_path = os.path.join(output_dir, file)
        cv2.imwrite(out_path, masked)
        print(f"Saved: {out_path}")

    print("Done")


def run_realsense_mode(mask_file, mask_dir, out_dir, serial, color_w, color_h, depth_w, depth_h, fps, max_frames):
    if rs is None:
        raise RuntimeError('pyrealsense2 is required for RealSense mode')

    os.makedirs(out_dir, exist_ok=True)
    # prepare mask iterator if mask_dir provided
    mask_paths = []
    if mask_dir is not None:
        if not os.path.isdir(mask_dir):
            raise RuntimeError(f'Mask dir not found: {mask_dir}')
        mask_paths = sorted([os.path.join(mask_dir, p) for p in os.listdir(mask_dir) if p.lower().endswith('.png') or p.lower().endswith('.jpg')])

    fixed_mask = None
    if mask_file is not None:
        if not os.path.exists(mask_file):
            raise RuntimeError(f'Mask file not found: {mask_file}')
        fixed_mask = load_mask(mask_file)

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    frame_idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            # convert to RGB just for consistent display if needed
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            # choose mask
            mask = None
            if fixed_mask is not None:
                mask = fixed_mask
            elif mask_paths:
                if frame_idx < len(mask_paths):
                    mask = load_mask(mask_paths[frame_idx])
                else:
                    mask = None

            if mask is not None and mask.shape[:2] != color.shape[:2]:
                mask = cv2.resize(mask, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

            masked = apply_mask_to_img(color, mask)

            # show
            cv2.imshow('Color (BGR)', color)
            if mask is not None:
                # visualize mask as 3-channel for display
                mvis = (mask * 255).astype(np.uint8)
                mvis = cv2.cvtColor(mvis, cv2.COLOR_GRAY2BGR)
                cv2.imshow('Mask', mvis)
                cv2.imshow('Masked', masked)
            else:
                cv2.imshow('Masked (no mask)', masked)

            out_path = os.path.join(out_dir, f'{frame_idx:06d}.png')
            cv2.imwrite(out_path, masked)

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if (max_frames is not None) and (frame_idx >= max_frames):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description='Mask/frame alignment verification (folder or RealSense stream)')
    p.add_argument('--mode', choices=['dir', 'realsense'], default='realsense')
    p.add_argument('--input_dir', type=str, default=None, help='Input images directory (dir mode)')
    p.add_argument('--mask_dir', type=str, default=None, help='Masks directory (dir mode or stream sequential masks)')
    p.add_argument('--mask_file', type=str, default="/home/khaled/FoundationPose/masks/000040.png", help='Single fixed mask file (stream mode)')
    p.add_argument('--output_dir', type=str, default=None, help='Output directory')
    # realsense options
    p.add_argument('--rs_serial', type=str, default=None)
    p.add_argument('--rs_color_w', type=int, default=960)
    p.add_argument('--rs_color_h', type=int, default=540)
    p.add_argument('--rs_depth_w', type=int, default=1280)
    p.add_argument('--rs_depth_h', type=int, default=720)
    p.add_argument('--rs_fps', type=int, default=30)
    p.add_argument('--max_frames', type=int, default=None)

    args = p.parse_args()

    if args.mode == 'dir':
        if args.input_dir is None or args.mask_dir is None:
            raise RuntimeError('For dir mode, --input_dir and --mask_dir are required')
        out = args.output_dir or os.path.join(os.path.dirname(args.input_dir), 'masked_output')
        run_dir_mode(args.input_dir, args.mask_dir, out)
    else:
        out = args.output_dir or os.path.join(os.getcwd(), 'masked_output_stream')
        run_realsense_mode(args.mask_file, args.mask_dir, out, args.rs_serial,
                           args.rs_color_w, args.rs_color_h, args.rs_depth_w, args.rs_depth_h,
                           args.rs_fps, args.max_frames)


if __name__ == '__main__':
    main()
