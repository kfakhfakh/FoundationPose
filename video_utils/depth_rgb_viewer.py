import argparse
import glob
import os
import time
import cv2
import numpy as np


def normalize_depth_for_display(depth, vmin=None, vmax=None, clip_percentiles=(2, 98)):
    d = depth.astype(np.float32)
    mask = d > 0
    if not np.any(mask):
        return np.zeros_like(d, dtype=np.uint8)

    valid = d[mask]
    if vmin is None or vmax is None:
        lo, hi = np.percentile(valid, clip_percentiles)
    else:
        lo, hi = vmin, vmax
    if hi <= lo:
        hi = lo + 1.0
    out = np.zeros_like(d, dtype=np.uint8)
    scaled = (d - lo) / (hi - lo)
    scaled = (scaled * 255.0).clip(0, 255)
    out[mask] = scaled[mask].astype(np.uint8)
    return out


def depth_to_colormap(depth, colormap=cv2.COLORMAP_JET, vmin=None, vmax=None):
    norm = normalize_depth_for_display(depth, vmin, vmax)
    cm = cv2.applyColorMap(norm, colormap)
    return cm


def on_mouse(event, x, y, flags, data):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    depth = data['depth']
    left_w = data.get('left_width', depth.shape[1])
    overlay = data.get('overlay', False)

    # If overlay is False, right half is depth_color and left is RGB
    if overlay:
        dx = x
        dy = y
    else:
        # click in combined image: if click is on right half, map to depth
        if x < left_w:
            print('Click is on RGB pane — switch to depth pane or toggle overlay')
            return
        dx = x - left_w
        dy = y

    h, w = depth.shape[:2]
    if dx < 0 or dx >= w or dy < 0 or dy >= h:
        print('Out of bounds')
        return

    val = int(depth[dy, dx])
    if val == 0:
        print(f'Pixel ({dx},{dy}): no depth')
    else:
        print(f'Pixel ({dx},{dy}): raw={val}  (units depend on source)')


def make_display(rgb, depth, resize=None, colormap=cv2.COLORMAP_JET, overlay=False, vmin=None, vmax=None):
    # rgb: BGR uint8 HxWx3
    # depth: single channel (uint8 or uint16)
    if rgb is None or depth is None:
        return None

    if resize is not None:
        rw, rh = resize
        rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
        depth_disp = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:
        depth_disp = depth

    depth_color = depth_to_colormap(depth_disp, colormap, vmin, vmax)

    if overlay:
        # ensure same type
        rgb_vis = rgb.copy()
        depth_vis = depth_color
        blended = cv2.addWeighted(rgb_vis, 0.6, depth_vis, 0.4, 0)
        left = blended
        right = depth_color
    else:
        left = rgb
        right = depth_color

    combined = np.hstack([left, right])
    return combined, depth_disp


def parse_args():
    p = argparse.ArgumentParser(description='Visualize RGB and depth images (supports 16-bit depth). Provide either --dataset or --rgb/--depth')
    p.add_argument('--dataset', '-D', default="C:\\Users\\kfakh\\OneDrive\\Desktop\\falku\\FoundationPose\\video_utils\\ref_views_1\ob_0000001", help='Path to dataset folder containing rgb/ and depth/ subfolders (default: ref_views_1 next to script)')
    p.add_argument('--rgb', '-r', help='Path to single RGB image (png/jpg)')
    p.add_argument('--depth', '-d', help='Path to single depth image (png, keep as saved 16-bit if available)')
    p.add_argument('--resize', default=None, help='Resize display as WIDTHxHEIGHT (e.g. 1280x720)')
    p.add_argument('--colormap', default='JET', help='Colormap name: JET, HOT, BONE, etc.')
    p.add_argument('--overlay', action='store_true', help='Show overlay (depth on top of RGB) on left side')
    p.add_argument('--save-on-s', action='store_true', help='Allow pressing s to save a screenshot of the combined view')
    p.add_argument('--play', action='store_true', help='Start playback of frames when using --dataset')
    return p.parse_args()


def colormap_name_to_code(name):
    name = name.upper()
    mapping = {
        'AUTUMN': cv2.COLORMAP_AUTUMN,
        'BONE': cv2.COLORMAP_BONE,
        'JET': cv2.COLORMAP_JET,
        'WINTER': cv2.COLORMAP_WINTER,
        'RAINBOW': cv2.COLORMAP_RAINBOW,
        'OCEAN': cv2.COLORMAP_OCEAN,
        'SUMMER': cv2.COLORMAP_SUMMER,
        'SPRING': cv2.COLORMAP_SPRING,
        'COOL': cv2.COLORMAP_COOL,
        'HSV': cv2.COLORMAP_HSV,
        'PINK': cv2.COLORMAP_PINK,
        'HOT': cv2.COLORMAP_HOT,
    }
    return mapping.get(name, cv2.COLORMAP_JET)


def main():
    args = parse_args()

    dataset = args.dataset
    rgb_list = []
    depth_list = []

    # If dataset is provided (or default), prefer it; otherwise require rgb+depth
    if dataset:
        if not os.path.isdir(dataset):
            print('Dataset folder not found:', dataset)
            return
        rgb_dir = os.path.join(dataset, 'rgb')
        depth_dir = os.path.join(dataset, 'depth')
        if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
            print('Dataset must contain rgb/ and depth/ subfolders')
            return

        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*')))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*')))
        rgb_map = {os.path.splitext(os.path.basename(p))[0]: p for p in rgb_files}
        depth_map = {os.path.splitext(os.path.basename(p))[0]: p for p in depth_files}
        common = sorted(set(rgb_map.keys()) & set(depth_map.keys()))
        if not common:
            print('No matching filenames between rgb/ and depth/')
            return
        rgb_list = [rgb_map[k] for k in common]
        depth_list = [depth_map[k] for k in common]
    else:
        if not args.rgb or not args.depth:
            print('Either --dataset or both --rgb and --depth must be provided')
            return
        if not os.path.exists(args.rgb) or not os.path.exists(args.depth):
            print('Provided files do not exist')
            return
        rgb_list = [args.rgb]
        depth_list = [args.depth]

    total = len(rgb_list)
    idx = 0

    def load_pair(i):
        rgb = cv2.imread(rgb_list[i], cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_list[i], cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise RuntimeError(f'Failed to load pair: {rgb_list[i]} / {depth_list[i]}')
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        return rgb, depth

    rgb, depth = load_pair(idx)

    if args.resize:
        try:
            rw, rh = args.resize.split('x')
            resize = (int(rw), int(rh))
        except Exception:
            resize = None
    else:
        resize = None

    cmap = colormap_name_to_code(args.colormap)
    overlay = args.overlay

    win = 'Depth-RGB Viewer'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    show_save_hint = args.save_on_s
    counter = 0
    play = args.play
    playback_delay = 100

    while True:
        combined, depth_disp = make_display(rgb, depth, resize, cmap, overlay)
        if combined is None:
            break

        status = f'[{idx+1}/{total}] '
        if dataset:
            status += os.path.basename(rgb_list[idx])
        cv2.putText(combined, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(win, combined)
        left_w = combined.shape[1] // 2
        data = {'depth': depth_disp, 'left_width': left_w, 'overlay': overlay}
        cv2.setMouseCallback(win, on_mouse, data)

        if play:
            key = cv2.waitKey(playback_delay) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord('o'):
            overlay = not overlay
        elif key == ord('s') and show_save_hint:
            out_path = f'depth_rgb_snapshot_{counter:03d}.png'
            cv2.imwrite(out_path, combined)
            print('Saved', out_path)
            counter += 1
        elif key == ord('n') or key == ord('d') or key == 83:
            idx = (idx + 1) % total
            rgb, depth = load_pair(idx)
        elif key == ord('p') or key == ord('a') or key == 81:
            idx = (idx - 1) % total
            rgb, depth = load_pair(idx)
        elif key == ord(' '):
            play = not play
        elif key == ord('f'):
            playback_delay = max(10, playback_delay - 10)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
