# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import argparse
import time
import cv2
import numpy as np
import trimesh
import torch

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(code_dir, 'mycpp', 'build'))
from Utils import *
from estimater import *
from datareader import *


def setup_torch(device_id: int):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Please run on a machine with CUDA drivers and a compatible GPU.')
    torch.cuda.set_device(device_id)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


def create_video_writer(output_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


class VideoFileReader:
    def __init__(self, video_file, depth_dir=None, mask_dir=None, cam_K=None, shorter_side=None, zfar=np.inf):
        self.video_file = video_file
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.cam_K = cam_K
        self.zfar = zfar
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise RuntimeError(f'Unable to open video file: {video_file}')

        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.id_strs = [f'{i:06d}' for i in range(self.length)]
        self.downscale = 1.0
        self.current_index = 0

        if shorter_side is not None:
            self.downscale = float(shorter_side) / min(self.H, self.W)
            self.H = int(self.H * self.downscale)
            self.W = int(self.W * self.downscale)
            self.cam_K[:2] *= self.downscale

    def __len__(self):
        return self.length

    def _read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f'Unable to read video frame at index {self.current_index}')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.downscale != 1.0:
            frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        self.current_index += 1
        return frame

    def get_color(self, i):
        if i != self.current_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.current_index = i
        return self._read_frame()

    def get_depth(self, i):
        if self.depth_dir is None:
            raise RuntimeError('Depth directory is required for video file inference')
        frame_name = os.path.basename(self.id_strs[i])
        for ext in ['png', 'jpg', 'jpeg', 'npy']:
            depth_path = os.path.join(self.depth_dir, f'{frame_name}.{ext}')
            if os.path.exists(depth_path):
                if ext == 'npy':
                    depth = np.load(depth_path)
                else:
                    depth = cv2.imread(depth_path, -1)
                    if depth is None:
                        continue
                    if depth.ndim == 3:
                        depth = depth[..., 0]
                    if depth.dtype not in (np.float32, np.float64):
                        depth = depth.astype(np.float32)
                        if depth.max() > 1000:
                            depth = depth / 1000.0
                if self.downscale != 1.0:
                    depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                depth[(depth < 0.001) | (depth >= self.zfar)] = 0
                return depth
        raise RuntimeError(f'Depth file not found for frame {i} in {self.depth_dir}')

    def get_mask(self, i):
        if self.mask_dir is None:
            return np.ones((self.H, self.W), dtype=np.uint8)
        frame_name = os.path.basename(self.id_strs[i])
        for ext in ['png', 'jpg', 'jpeg']:
            mask_path = os.path.join(self.mask_dir, f'{frame_name}.{ext}')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, -1)
                if mask is None:
                    continue
                if mask.ndim == 3:
                    mask = mask[..., 0]
                mask = (mask > 0).astype(np.uint8)
                if self.downscale != 1.0:
                    mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                return mask
        return np.ones((self.H, self.W), dtype=np.uint8)

    def release(self):
        self.cap.release()


def inference_loop(reader, mesh, args):
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext(args.device)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                         refiner=refiner, debug_dir=args.debug_dir, debug=int(args.visualize), glctx=glctx)
    est.to_device(f'cuda:{args.device}')

    if hasattr(reader, 'cam_K'):
        K = reader.cam_K
    else:
        K = reader.K

    writer = None
    if args.save_video:
        os.makedirs(args.debug_dir, exist_ok=True)
        writer = create_video_writer(os.path.join(args.debug_dir, 'inference_video.mp4'), reader.W, reader.H,
                                     int(getattr(reader, 'fps', 30)))

    for i in range(len(reader)):
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if args.depth_scale != 1.0:
            depth = depth.astype(np.float32, copy=False) * args.depth_scale

        if i == 0:
            mask = reader.get_mask(i).astype(bool)
            torch.cuda.synchronize()
            gpu_start = time.time()
            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            torch.cuda.synchronize()
            gpu_start = time.time()
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

        torch.cuda.synchronize()
        gpu_time = time.time() - gpu_start
        logging.info(f'Frame {i+1}/{len(reader)} GPU inference time: {gpu_time:.3f} sec')

        os.makedirs(f'{args.debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{args.debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0,
                             is_input_rgb=True)

        if args.visualize:
            cv2.imshow('FoundationPose inference', vis[..., ::-1])
            cv2.waitKey(1)

        if writer is not None:
            writer.write(vis[..., ::-1])

    if writer is not None:
        writer.release()
    if isinstance(reader, VideoFileReader):
        reader.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FoundationPose optimized inference with video save')
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--video_dir', type=str, default=f'{code_dir}/demo_data/mustard0',
                        help='Folder with rgb/, depth/, masks/ and cam_K.txt')
    parser.add_argument('--video_file', type=str, default=None, help='Video file for color frames')
    parser.add_argument('--depth_dir', type=str, default=None, help='Optional depth frame directory for video_file mode')
    parser.add_argument('--mask_dir', type=str, default=None, help='Optional mask frame directory for video_file mode')
    parser.add_argument('--cam_K_file', type=str, default=None, help='Camera intrinsic matrix file for video_file mode')
    parser.add_argument('--shorter_side', type=int, default=None, help='Resize shorter side to this value')
    parser.add_argument('--zfar', type=float, default=np.inf, help='Depth far clipping plane')
    parser.add_argument('--depth_scale', type=float, default=1.0,
                        help='Multiplier applied to depth values before inference')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--visualize', type=int, default=1, help='Show output frames in a window')
    parser.add_argument('--save_video', type=int, default=1, help='Save output visualization as mp4')
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id to use for inference')
    args = parser.parse_args()

    if args.depth_scale <= 0:
        raise RuntimeError('--depth_scale must be > 0')

    setup_torch(args.device)

    if args.video_dir is None and args.video_file is None:
        raise RuntimeError('Please provide either --video_dir or --video_file')

    if args.video_dir is not None:
        reader = YcbineoatReader(args.video_dir, shorter_side=args.shorter_side, zfar=args.zfar)
    else:
        cam_K = load_camera_matrix(args.cam_K_file)
        reader = VideoFileReader(args.video_file, depth_dir=args.depth_dir, mask_dir=args.mask_dir,
                                 cam_K=cam_K, shorter_side=args.shorter_side, zfar=args.zfar)

    mesh = trimesh.load(args.mesh_file)
    inference_loop(reader, mesh, args)
