# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from Utils import *
from estimater import *
from datareader import *
import argparse, os, cv2, numpy as np, trimesh, imageio, torch, time
from concurrent.futures import ThreadPoolExecutor


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
    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    self.id_strs = [f'{i:06d}' for i in range(self.length)]
    self.downscale = 1.0
    if shorter_side is not None:
      self.downscale = float(shorter_side) / min(self.H, self.W)
      self.H = int(self.H * self.downscale)
      self.W = int(self.W * self.downscale)
      self.cam_K[:2] *= self.downscale

  def __len__(self):
    return self.length

  def _read_frame(self, frame_idx):
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = self.cap.read()
    if not ret:
      raise RuntimeError(f'Unable to read frame {frame_idx}')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if self.downscale != 1.0:
      frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
    return frame

  def get_color(self, i):
    return self._read_frame(i)

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
            depth = depth[...,0]
          if depth.dtype != np.float32 and depth.dtype != np.float64:
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
      return None
    frame_name = os.path.basename(self.id_strs[i])
    for ext in ['png', 'jpg', 'jpeg']:
      mask_path = os.path.join(self.mask_dir, f'{frame_name}.{ext}')
      if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, -1)
        if mask is None:
          continue
        if mask.ndim == 3:
          mask = mask[...,0]
        mask = (mask > 0).astype(np.uint8)
        if self.downscale != 1.0:
          mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return mask
    return None#np.ones((self.H, self.W), dtype=np.uint8)

  def release(self):
    self.cap.release()


def load_camera_matrix(cam_K_file):
  if cam_K_file is None:
    raise RuntimeError('Camera intrinsic file is required for video file mode')
  K = np.loadtxt(cam_K_file).reshape(3,3)
  return K


def load_mesh_file(mesh_path):
  mesh = trimesh.load(mesh_path, force='mesh', process=False)
  if isinstance(mesh, trimesh.Scene):
    geometries = [geometry for geometry in mesh.geometry.values()]
    if len(geometries) == 0:
      raise RuntimeError(f'Unable to load mesh geometry from {mesh_path}')
    mesh = trimesh.util.concatenate(geometries)
  return mesh


def get_object_frame(mesh):
  try:
    return trimesh.bounds.oriented_bounds(mesh)
  except Exception:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
      raise
    to_origin = np.eye(4)
    to_origin[:3, 3] = -bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    return to_origin, extents


def make_video_writer(output_path, width, height, fps):
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def load_frame(reader, i):
  if isinstance(reader, VideoFileReader):
    cap = cv2.VideoCapture(reader.video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    cap.release()
    if not ret:
      raise RuntimeError(f'Unable to open frame {i} from {reader.video_file}')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if reader.downscale != 1.0:
      frame = cv2.resize(frame, (reader.W, reader.H), interpolation=cv2.INTER_LINEAR)
    color = frame
  else:
    color = reader.get_color(i)

  depth = reader.get_depth(i)
  try:
    mask = reader.get_mask(i)
  except Exception as e:
    logging.warning(f'Frame {i}: unable to load mask, continuing without mask. Error: {e}')
    mask = None
  return {'color': color, 'depth': depth, 'mask': mask}


def extract_translation_and_rotation(pose_matrix):
  """
  Extract translation vector and rotation angles (Euler angles) from a 4x4 pose matrix.
  
  Args:
    pose_matrix: 4x4 numpy array or torch tensor representing the pose
  
  Returns:
    translation: 3D translation vector (x, y, z) in meters
    euler_angles: Rotation as Euler angles (roll, pitch, yaw) in degrees
  """
  if isinstance(pose_matrix, torch.Tensor):
    pose_matrix = pose_matrix.cpu().numpy()
  
  # Extract translation (last column, first 3 rows)
  translation = pose_matrix[:3, 3]
  
  # Extract rotation matrix (top-left 3x3)
  rotation_matrix = pose_matrix[:3, :3]
  
  # Convert rotation matrix to Euler angles (ZYX convention)
  # Roll (rotation around X-axis)
  roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
  
  # Pitch (rotation around Y-axis)
  pitch = np.arcsin(-rotation_matrix[2, 0])
  
  # Yaw (rotation around Z-axis)
  yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
  
  # Convert from radians to degrees
  euler_angles = np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])
  
  return translation, euler_angles


def inference_loop(reader, mesh, args):
  to_origin, extents = get_object_frame(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext(args.device)
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=args.debug_dir, debug=args.debug, glctx=glctx)
  est.to_device(f'cuda:{args.device}')

  if isinstance(reader, VideoFileReader):
    K = reader.cam_K
  else:
    K = reader.K

  if args.save_video:
    os.makedirs(args.debug_dir, exist_ok=True)
    writer = make_video_writer(os.path.join(args.debug_dir, 'inference_video.mp4'), reader.W, reader.H, getattr(reader, 'fps', 30))
  else:
    writer = None

  executor = ThreadPoolExecutor(max_workers=1)
  next_future = executor.submit(load_frame, reader, 0)

  for i in range(len(reader)):
    logging.info(f'Frame {i+1}/{len(reader)}')
    torch.cuda.synchronize()
    frame_start = time.time()

    frame = next_future.result()
    if i + 1 < len(reader):
      next_future = executor.submit(load_frame, reader, i + 1)

    color = frame['color']
    depth = frame['depth']
    if args.depth_scale != 1.0:
      depth = depth.astype(np.float32, copy=False) * args.depth_scale

    mask = frame['mask']
    has_mask = (mask is not None) and np.any(mask)

    gpu_start = time.time()
    if args.inference_mode == 'frame_registration':
      # Robust mode: re-register each frame when a mask is available.
      if has_mask:
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      elif i == 0 or est.pose_last is None:
        # If no external mask is available on initialization, bootstrap with valid depth region.
        init_mask = (depth >= 0.001).astype(np.uint8)
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=init_mask, iteration=args.est_refine_iter)
      else:
        pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
    else:
      # Fast mode: initialize once, then track.
      if i == 0 or est.pose_last is None:
        init_mask = mask if has_mask else (depth >= 0.001).astype(np.uint8)
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=init_mask, iteration=args.est_refine_iter)
      else:
        pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

    torch.cuda.synchronize()
    gpu_time = time.time() - gpu_start
    frame_time = time.time() - frame_start
    logging.info(f'Frame {i+1} inference time: {frame_time:.3f} sec (GPU compute: {gpu_time:.3f} sec)')

    # Print pose information if requested
    if args.print_pose_info:
      translation, euler_angles = extract_translation_and_rotation(pose)
      print(f"\n--- Frame {i} ---")
      print(f"Translation (meters): x={translation[0]:.6f}, y={translation[1]:.6f}, z={translation[2]:.6f}")
      print(f"Distance from camera: {np.linalg.norm(translation):.6f} meters")
      print(f"Rotation (Euler angles in degrees): roll={euler_angles[0]:.2f}°, pitch={euler_angles[1]:.2f}°, yaw={euler_angles[2]:.2f}°")

    os.makedirs(f'{args.debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{args.debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)

    if args.debug >= 1:
      cv2.imshow('FoundationPose inference', vis[..., ::-1])
      cv2.waitKey(1)

    if args.debug >= 2:
      os.makedirs(f'{args.debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{args.debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    if writer is not None:
      writer.write(vis[..., ::-1])

  if writer is not None:
    writer.release()
  if isinstance(reader, VideoFileReader):
    reader.release()
  executor.shutdown(wait=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='FoundationPose video inference')
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--mesh_scale', type=float, default=1.0, help='Uniform scale factor applied to mesh geometry before inference')
  parser.add_argument('--video_dir', type=str, default=f'{code_dir}/demo_data/mustard0', help='Folder with rgb/, depth/, masks/ and cam_K.txt')
  parser.add_argument('--video_file', type=str, default=None, help='Video file for color frames')
  parser.add_argument('--depth_dir', type=str, default=None, help='Optional depth frame directory for video_file mode')
  parser.add_argument('--mask_dir', type=str, default=None, help='Optional mask frame directory for video_file mode')
  parser.add_argument('--cam_K_file', type=str, default=None, help='Camera intrinsic matrix file for video_file mode')
  parser.add_argument('--shorter_side', type=int, default=None, help='Resize shorter side to this value')
  parser.add_argument('--zfar', type=float, default=np.inf, help='Depth far clipping plane')
  parser.add_argument('--depth_scale', type=float, default=1.0, help='Multiplier applied to depth values before inference')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--inference_mode', type=str, choices=['frame_registration', 'fast'], default='fast', help='frame_registration: register each frame when mask exists (robust); fast: initialize once then track (faster)')
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/output/debug')
  parser.add_argument('--save_video', type=int, default=0, help='Save output visualization as mp4')
  parser.add_argument('--print_pose_info', type=int, default=0, help='Print translation and rotation info for each frame to terminal')
  parser.add_argument('--device', type=int, default=0, help='CUDA device id to use for inference')
  args = parser.parse_args()

  if args.mesh_scale <= 0:
    raise RuntimeError('--mesh_scale must be > 0')
  if args.depth_scale <= 0:
    raise RuntimeError('--depth_scale must be > 0')

  set_logging_format()
  set_seed(0)

  if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available. Please run on a machine with CUDA drivers and a compatible GPU.')
  torch.cuda.set_device(args.device)
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  torch.backends.cuda.matmul.allow_tf32 = True

  if args.video_dir is None and args.video_file is None:
    raise RuntimeError('Please provide either --video_dir or --video_file')

  if args.video_dir is not None:
    reader = YcbineoatReader(args.video_dir, shorter_side=args.shorter_side, zfar=args.zfar)
  else:
    cam_K = load_camera_matrix(args.cam_K_file)
    reader = VideoFileReader(args.video_file, depth_dir=args.depth_dir, mask_dir=args.mask_dir, cam_K=cam_K, shorter_side=args.shorter_side, zfar=args.zfar)

  mesh = load_mesh_file(args.mesh_file)
  if args.mesh_scale != 1.0:
    if not hasattr(mesh, 'apply_scale'):
      raise RuntimeError(f'Loaded mesh type does not support scaling: {type(mesh)}')
    mesh.apply_scale(args.mesh_scale)
  inference_loop(reader, mesh, args)
