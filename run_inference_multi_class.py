# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from Utils import *
from estimater import *
import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MESH_EXTS = {'.obj', '.ply', '.stl', '.glb', '.gltf'}


def normalize_name(name):
  return re.sub(r'[^a-z0-9]+', '', str(name).lower())


def make_safe_name(name):
  safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name).strip())
  return safe or 'object'


def strip_known_image_extension(name):
  text = str(name).strip()
  suffix = Path(text).suffix.lower()
  if suffix in IMAGE_EXTS:
    return Path(text).stem
  return text


def setup_torch(device_id):
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


def load_camera_matrix(cam_K_file):
  if cam_K_file is None:
    raise RuntimeError('Camera intrinsic matrix file is required')
  return np.loadtxt(cam_K_file).reshape(3, 3)


def load_mesh_file(mesh_path):
  mesh = trimesh.load(mesh_path, force='mesh', process=False)
  if isinstance(mesh, trimesh.Scene):
    geometries = [geometry for geometry in mesh.geometry.values()]
    if len(geometries) == 0:
      raise RuntimeError(f'Unable to load mesh geometry from {mesh_path}')
    mesh = trimesh.util.concatenate(geometries)
  return mesh


def create_video_writer(output_path, width, height, fps):
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def color_from_name(name):
  digest = hashlib.md5(str(name).encode('utf-8')).digest()
  return tuple(int(64 + value % 192) for value in digest[:3])


class FolderSceneReader:
  def __init__(self, scene_dir, cam_K_file=None, shorter_side=None, zfar=np.inf,
               rgb_dir_name='rgb', depth_dir_name='depth', masks_dir_name='masks'):
    self.scene_dir = scene_dir
    self.rgb_dir = os.path.join(scene_dir, rgb_dir_name)
    self.depth_dir = os.path.join(scene_dir, depth_dir_name)
    self.masks_dir = os.path.join(scene_dir, masks_dir_name)
    self.zfar = zfar
    self.fps = 30

    if not os.path.isdir(self.rgb_dir):
      raise RuntimeError(f'RGB folder not found: {self.rgb_dir}')
    if not os.path.isdir(self.depth_dir):
      raise RuntimeError(f'Depth folder not found: {self.depth_dir}')

    self.rgb_files = sorted([path for path in Path(self.rgb_dir).iterdir() if path.suffix.lower() in IMAGE_EXTS])
    if len(self.rgb_files) == 0:
      raise RuntimeError(f'No RGB images found in {self.rgb_dir}')

    self.id_strs = [path.stem for path in self.rgb_files]
    self.frame_id_norms = {normalize_name(frame_id) for frame_id in self.id_strs}

    first_rgb = cv2.imread(str(self.rgb_files[0]), cv2.IMREAD_COLOR)
    if first_rgb is None:
      raise RuntimeError(f'Unable to read first RGB frame: {self.rgb_files[0]}')
    self.H, self.W = first_rgb.shape[:2]

    if cam_K_file is None:
      cam_K_file = os.path.join(scene_dir, 'cam_K.txt')
    self.cam_K = load_camera_matrix(cam_K_file)

    self.downscale = 1.0
    if shorter_side is not None:
      self.downscale = float(shorter_side) / min(self.H, self.W)
      self.H = int(round(self.H * self.downscale))
      self.W = int(round(self.W * self.downscale))
      self.cam_K[:2] *= self.downscale

    self.mask_files = []
    self.mask_index = {}
    self.object_name_lookup = {}
    self.object_to_paths = defaultdict(list)
    self._build_mask_index()

  def __len__(self):
    return len(self.rgb_files)

  def _build_mask_index(self):
    if not os.path.isdir(self.masks_dir):
      return

    for mask_path in sorted(Path(self.masks_dir).rglob('*')):
      if mask_path.suffix.lower() not in IMAGE_EXTS:
        continue
      self.mask_files.append(mask_path)

      # Naming convention: masks/obj_000005.png -> object name is obj_000005.
      object_name_clean = strip_known_image_extension(mask_path.stem)
      object_norm = normalize_name(object_name_clean)
      if not object_norm:
        continue

      if object_norm not in self.object_name_lookup:
        self.object_name_lookup[object_norm] = object_name_clean
      self.object_to_paths[object_norm].append(mask_path)

  def discover_object_names(self):
    object_names = []
    for object_norm in sorted(self.object_name_lookup):
      object_names.append(self.object_name_lookup[object_norm])
    return object_names

  def get_color(self, i):
    color = cv2.imread(str(self.rgb_files[i]), cv2.IMREAD_COLOR)
    if color is None:
      raise RuntimeError(f'Unable to read RGB frame: {self.rgb_files[i]}')
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    if self.downscale != 1.0:
      color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
    return color

  def get_depth(self, i):
    frame_id = self.id_strs[i]
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy']:
      depth_path = os.path.join(self.depth_dir, f'{frame_id}{ext}')
      if not os.path.exists(depth_path):
        continue
      if ext == '.npy':
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
    raise RuntimeError(f'Depth file not found for frame {frame_id} in {self.depth_dir}')

  def _read_mask_file(self, mask_path):
    mask = cv2.imread(str(mask_path), -1)
    if mask is None:
      return None
    if mask.ndim == 3:
      mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    if self.downscale != 1.0:
      mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
    return mask

  def get_mask(self, i, object_name):
    object_norm = normalize_name(object_name)
    paths = self.object_to_paths.get(object_norm, [])
    if len(paths) == 0:
      return None

    # One mask per object naming convention.
    mask_path = paths[0]
    return self._read_mask_file(mask_path)


def build_model_index(models_dir):
  if not os.path.isdir(models_dir):
    raise RuntimeError(f'Models folder not found: {models_dir}')

  model_index = {}
  for path in Path(models_dir).rglob('*'):
    if not path.is_file() or path.suffix.lower() not in MESH_EXTS:
      continue
    object_norm = normalize_name(path.stem)
    if object_norm and object_norm not in model_index:
      model_index[object_norm] = path
  return model_index


def resolve_mesh_path(model_index, object_name):
  object_norm = normalize_name(object_name)
  return model_index.get(object_norm)


def load_object_meshes(reader, models_dir, mesh_scale):
  model_index = build_model_index(models_dir)
  object_names = reader.discover_object_names()
  if len(object_names) == 0:
    raise RuntimeError(f'No object masks were discovered in {reader.masks_dir}')

  logging.info(f'Detected mask objects ({len(object_names)}): {sorted(object_names)}')

  selected_objects = []
  missing_models = []
  for object_name in object_names:
    mesh_path = resolve_mesh_path(model_index, object_name)
    if mesh_path is None:
      missing_models.append(object_name)
      continue
    mesh = load_mesh_file(mesh_path)
    if mesh_scale != 1.0:
      if not hasattr(mesh, 'apply_scale'):
        raise RuntimeError(f'Loaded mesh type does not support scaling: {type(mesh)}')
      mesh.apply_scale(mesh_scale)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
    selected_objects.append({
      'name': object_name,
      'safe_name': make_safe_name(object_name),
      'mesh_path': str(mesh_path),
      'mesh': mesh,
      'to_origin': to_origin,
      'bbox': bbox,
    })

  if len(selected_objects) == 0:
    raise RuntimeError(
      'No object meshes matched the masks. '
      f'Missing models for: {", ".join(missing_models) if missing_models else "unknown"}'
    )

  if len(missing_models) > 0:
    logging.info(f'Skipped masks without matching models: {missing_models}')

  logging.info('Resolved object->model mapping:')
  for obj in selected_objects:
    logging.info(f"  {obj['name']} -> {obj['mesh_path']}")

  return selected_objects


def build_estimators(objects, args):
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  estimators = []
  estimator_debug_dir = None
  if args.debug >= 2:
    estimator_debug_dir = os.path.join(args.debug_dir, '_estimator_internal')
    os.makedirs(estimator_debug_dir, exist_ok=True)
  for obj in objects:
    estimator = FoundationPose(
      model_pts=obj['mesh'].vertices,
      model_normals=obj['mesh'].vertex_normals,
      mesh=obj['mesh'],
      scorer=scorer,
      refiner=refiner,
      debug_dir=estimator_debug_dir,
      debug=args.debug,
    )
    estimator.to_device(f'cuda:{args.device}')
    estimators.append(estimator)
  return estimators


def inference_loop(reader, objects, estimators, args):
  # Only run inference for objects that have a resolved model and estimator.
  if len(objects) != len(estimators):
    raise RuntimeError('Internal mismatch: objects and estimators must have the same length')

  active_pairs = [(obj, est) for obj, est in zip(objects, estimators) if obj.get('mesh_path')]
  if len(active_pairs) == 0:
    raise RuntimeError('No valid object-model pairs available for inference')

  logging.info(f'Running inference on {len(active_pairs)} resolved models only')

  frame_results = defaultdict(dict)
  output_pose_dir = os.path.join(args.debug_dir, 'ob_in_cam')
  if args.save_pose_txt:
    os.makedirs(output_pose_dir, exist_ok=True)

  if args.save_video:
    os.makedirs(args.debug_dir, exist_ok=True)
    writer = create_video_writer(
      os.path.join(args.debug_dir, 'inference_video.mp4'),
      reader.W,
      reader.H,
      getattr(reader, 'fps', 30),
    )
  else:
    writer = None

  live_view_enabled = bool(args.show_live)
  if live_view_enabled:
    try:
      cv2.startWindowThread()
      cv2.namedWindow('FoundationPose multi-object inference', cv2.WINDOW_NORMAL)
    except cv2.error:
      live_view_enabled = False
      logging.warning('Live preview is unavailable in this environment. Continuing without cv2.imshow.')
    except Exception:
      live_view_enabled = False
      logging.warning('Live preview could not be initialized. Continuing without cv2.imshow.')

  # Timing statistics
  timing_stats = defaultdict(list)
  total_start = time.time()

  for frame_idx in range(len(reader)):
    frame_start = time.time()
    frame_id = reader.id_strs[frame_idx]
    logging.info(f'\n{"=" * 70}')
    logging.info(f'Frame {frame_idx + 1}/{len(reader)}: {frame_id}')
    logging.info(f'{"=" * 70}')

    # Load frame data
    load_start = time.time()
    color = reader.get_color(frame_idx)
    color_time = time.time() - load_start
    timing_stats['load_color'].append(color_time)
    logging.info(f'  [LOAD] Color image: {color_time*1000:.2f}ms')

    depth_start = time.time()
    depth = reader.get_depth(frame_idx)
    depth_time = time.time() - depth_start
    timing_stats['load_depth'].append(depth_time)
    logging.info(f'  [LOAD] Depth image: {depth_time*1000:.2f}ms')

    if args.depth_scale != 1.0:
      depth = depth.astype(np.float32, copy=False) * args.depth_scale

    vis = color.copy()

    for object_info, estimator in active_pairs:
      object_name = object_info['name']
      pose = None

      # Get mask
      mask_start = time.time()
      mask = reader.get_mask(frame_idx, object_name)
      mask_time = time.time() - mask_start
      timing_stats[f'load_mask_{object_name}'].append(mask_time)
      logging.info(f'  [LOAD] Mask ({object_name}): {mask_time*1000:.2f}ms')

      if mask is None or mask.sum() == 0:
        logging.info(f'  [SKIP] {object_name}: No valid mask')
        continue

      if estimator.pose_last is None:
        # Registration (initialization)
        torch.cuda.synchronize()
        pose_start = time.time()
        pose = estimator.register(
          K=reader.cam_K,
          rgb=color,
          depth=depth,
          ob_mask=mask,
          iteration=args.est_refine_iter,
        )
        pose_time = time.time() - pose_start
        timing_stats[f'register_{object_name}'].append(pose_time)
        logging.info(f'  [REGISTER] {object_name}: {pose_time:.3f}s')

        if estimator.pose_last is None:
          centered_pose = torch.as_tensor(pose, device='cuda', dtype=torch.float) @ torch.linalg.inv(estimator.get_tf_to_centered_mesh())
          estimator.pose_last = centered_pose
      else:
        # Tracking
        torch.cuda.synchronize()
        pose_start = time.time()
        pose = estimator.track_one(
          rgb=color,
          depth=depth,
          K=reader.cam_K,
          iteration=args.track_refine_iter,
        )
        pose_time = time.time() - pose_start
        timing_stats[f'track_{object_name}'].append(pose_time)
        logging.info(f'  [TRACK] {object_name}: {pose_time:.3f}s')

      torch.cuda.synchronize()

      if pose is None:
        logging.info(f'  [SKIP] {object_name}: Pose estimation failed')
        continue

      # Save pose
      save_start = time.time()
      frame_results[frame_id][object_name] = pose.tolist()
      if args.save_pose_txt:
        pose_dir = os.path.join(output_pose_dir, object_info['safe_name'])
        os.makedirs(pose_dir, exist_ok=True)
        np.savetxt(os.path.join(pose_dir, f'{frame_id}.txt'), pose.reshape(4, 4))
      save_time = time.time() - save_start
      timing_stats[f'save_pose_{object_name}'].append(save_time)
      logging.info(f'  [SAVE] Pose ({object_name}): {save_time*1000:.2f}ms')

      # Visualize
      vis_start = time.time()
      center_pose = pose @ np.linalg.inv(object_info['to_origin'])
      line_color = color_from_name(object_name)
      vis = draw_posed_3d_box(reader.cam_K, img=vis, ob_in_cam=center_pose, bbox=object_info['bbox'], line_color=line_color)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.cam_K, thickness=3, transparency=0, is_input_rgb=True)
      vis_time = time.time() - vis_start
      timing_stats[f'visualize_{object_name}'].append(vis_time)
      logging.info(f'  [VIS] Draw ({object_name}): {vis_time*1000:.2f}ms')

    # Display/save frame
    display_start = time.time()
    if live_view_enabled:
      try:
        cv2.imshow('FoundationPose multi-object inference', vis[..., ::-1])
        cv2.waitKey(1)
      except cv2.error:
        live_view_enabled = False
        logging.warning('cv2.imshow failed during runtime. Live preview has been disabled.')

    if args.debug >= 2:
      track_vis_dir = os.path.join(args.debug_dir, 'track_vis')
      os.makedirs(track_vis_dir, exist_ok=True)
      imageio.imwrite(os.path.join(track_vis_dir, f'{frame_id}.png'), vis)

    if writer is not None:
      writer.write(vis[..., ::-1])

    display_time = time.time() - display_start
    timing_stats['display_frame'].append(display_time)
    logging.info(f'  [DISPLAY] Frame display/save: {display_time*1000:.2f}ms')

    frame_total = time.time() - frame_start
    timing_stats['total_per_frame'].append(frame_total)
    logging.info(f'  [TOTAL] Frame {frame_idx + 1} processed in {frame_total:.3f}s')

  if writer is not None:
    writer.release()

  if live_view_enabled:
    cv2.destroyAllWindows()

  if args.save_results_json:
    os.makedirs(args.debug_dir, exist_ok=True)
    result_path = os.path.join(args.debug_dir, 'poses.json')
    with open(result_path, 'w', encoding='utf-8') as ff:
      json.dump(frame_results, ff, indent=2)

  # Print timing summary
  total_time = time.time() - total_start
  print("\n" + "=" * 80)
  print("⏱️  INFERENCE TIMING SUMMARY")
  print("=" * 80)
  print(f"Total inference time: {total_time:.2f}s")
  print(f"Total frames processed: {len(reader)}")
  print(f"Average time per frame: {total_time / len(reader):.3f}s")
  print("\nDetailed timing breakdown:")
  print("-" * 80)

  for key in sorted(timing_stats.keys()):
    times = timing_stats[key]
    total = sum(times)
    avg = total / len(times) if times else 0
    if len(times) == 1:
      print(f"  {key:45s}: {times[0]:8.3f}s")
    else:
      min_t = min(times)
      max_t = max(times)
      print(f"  {key:45s}: Total={total:8.2f}s | Avg={avg:7.3f}s | Min={min_t:7.3f}s | Max={max_t:7.3f}s | Count={len(times)}")

  print("=" * 80 + "\n")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='FoundationPose multi-object video inference')
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--video_dir', type=str, default=f'{code_dir}/demo_data/mustard0', help='Folder with rgb/, depth/, masks/ and cam_K.txt')
  parser.add_argument('--models_dir', type=str, required=True, help='Folder containing the 3D object models')
  parser.add_argument('--cam_K_file', type=str, default=None, help='Override the camera intrinsic matrix file')
  parser.add_argument('--shorter_side', type=int, default=None, help='Resize shorter side to this value')
  parser.add_argument('--zfar', type=float, default=np.inf, help='Depth far clipping plane')
  parser.add_argument('--mesh_scale', type=float, default=1.0, help='Uniform scale factor applied to every model before inference')
  parser.add_argument('--depth_scale', type=float, default=1.0, help='Multiplier applied to depth values before inference')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--show_live', type=int, default=1, help='Show live OpenCV preview window')
  parser.add_argument('--save_video', type=int, default=0, help='Save output visualization as mp4')
  parser.add_argument('--save_pose_txt', type=int, default=0, help='Save per-frame pose text files under debug/ob_in_cam')
  parser.add_argument('--save_results_json', type=int, default=0, help='Save poses.json under debug_dir')
  parser.add_argument('--device', type=int, default=0, help='CUDA device id to use for inference')
  args = parser.parse_args()

  if args.mesh_scale <= 0:
    raise RuntimeError('--mesh_scale must be > 0')
  if args.depth_scale <= 0:
    raise RuntimeError('--depth_scale must be > 0')

  set_logging_format()
  set_seed(0)
  setup_torch(args.device)

  reader = FolderSceneReader(
    args.video_dir,
    cam_K_file=args.cam_K_file,
    shorter_side=args.shorter_side,
    zfar=args.zfar,
  )

  objects = load_object_meshes(reader, args.models_dir, args.mesh_scale)
  estimators = build_estimators(objects, args)
  inference_loop(reader, objects, estimators, args)
