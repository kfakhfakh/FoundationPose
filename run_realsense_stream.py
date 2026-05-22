# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from Utils import *
from estimater import *
import argparse, os, cv2, numpy as np, trimesh, imageio, torch, time

try:
  import pyrealsense2 as rs
except Exception:
  rs = None


class RealSenseStreamReader:
  def __init__(self, color_width=1280, color_height=720, depth_width=1280, depth_height=720,
               fps=30, shorter_side=None, zfar=np.inf, serial=None, cam_K=None, mask_file=None):
    if rs is None:
      raise RuntimeError('pyrealsense2 is not installed; RealSense streaming is unavailable')

    self.zfar = zfar
    self.fps = fps
    self.streaming = True
    self.frame_index = 0
    self.downscale = 1.0

    self.fixed_mask = None
    if mask_file is not None:
      if not os.path.exists(mask_file):
        raise RuntimeError(f'Mask file not found: {mask_file}')
      fixed = cv2.imread(mask_file, -1)
      if fixed is None:
        raise RuntimeError(f'Unable to read mask file: {mask_file}')
      if fixed.ndim == 3:
        fixed = fixed[..., 0]
      self.fixed_mask = (fixed > 0).astype(np.uint8)

    self.pipeline = rs.pipeline()
    self.config = rs.config()
    if serial:
      self.config.enable_device(serial)
    self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
    self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
    self.profile = self.pipeline.start(self.config)
    self.align = rs.align(rs.stream.color)

    device = self.profile.get_device()
    self.depth_scale = device.first_depth_sensor().get_depth_scale()

    color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    stream_K = np.array([[intr.fx, 0.0, intr.ppx],
                         [0.0, intr.fy, intr.ppy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    if cam_K is not None:
      self.K = np.asarray(cam_K, dtype=np.float64).reshape(3, 3).copy()
    else:
      self.K = stream_K

    self.width = intr.width
    self.height = intr.height
    if shorter_side is not None:
      self.downscale = float(shorter_side) / min(self.height, self.width)
      self.height = int(self.height * self.downscale)
      self.width = int(self.width * self.downscale)
      self.K[:2] *= self.downscale

    self.W = self.width
    self.H = self.height

  def read(self):
    frames = self.pipeline.wait_for_frames()
    aligned_frames = self.align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
      return None

    color = np.asanyarray(color_frame.get_data())
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale

    if self.downscale != 1.0:
      color = cv2.resize(color, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
      depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

    depth[(depth < 0.001) | (depth >= self.zfar)] = 0

    mask = None
    if self.fixed_mask is not None:
      mask = self.fixed_mask.copy()
      if self.downscale != 1.0:
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

    idx = self.frame_index
    self.frame_index += 1
    return {
      'color': color,
      'depth': depth,
      'mask': mask,
      'id_str': f'{idx:06d}',
    }

  def release(self):
    self.pipeline.stop()


def load_camera_matrix(cam_K_file):
  if cam_K_file is None:
    return None
  return np.loadtxt(cam_K_file).reshape(3, 3)


def make_video_writer(output_path, width, height, fps):
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def extract_translation_and_rotation(pose_matrix):
  if isinstance(pose_matrix, torch.Tensor):
    pose_matrix = pose_matrix.cpu().numpy()

  translation = pose_matrix[:3, 3]
  rotation_matrix = pose_matrix[:3, :3]
  roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
  pitch = np.arcsin(-rotation_matrix[2, 0])
  yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
  euler_angles = np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])
  return translation, euler_angles


def verify_mesh_texture(mesh, mesh_file):
  """Log texture/color information for loaded mesh."""
  logging.info(f"Mesh file: {mesh_file}")
  
  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    logging.info("✓ TEXTURE LOADED: Mesh has texture (TextureVisuals)")
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
      tex_size = mesh.visual.material.image.size
      logging.info(f"  Texture size: {tex_size[0]}x{tex_size[1]} pixels")
    if hasattr(mesh.visual, 'uv'):
      logging.info(f"  UV coordinates: {mesh.visual.uv.shape[0]} vertices")
  elif hasattr(mesh, 'visual') and mesh.visual.vertex_colors is not None:
    logging.info("✓ VERTEX COLORS LOADED: Mesh has per-vertex colors")
    logging.info(f"  Colored vertices: {mesh.visual.vertex_colors.shape[0]}")
  else:
    logging.warning("⚠ NO TEXTURE/COLORS: Mesh will render with default gray (128,128,128)")
    logging.warning("  To use textures, ensure:")
    logging.warning("    - Texture files (.png/.jpg) are in same directory as mesh")
    logging.warning("    - .mtl file references correct texture filenames")
    logging.warning("    - Format supports textures (.obj+.mtl, .glb, .gltf)")


def main():
  parser = argparse.ArgumentParser(description='FoundationPose RealSense stream inference')
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/dataset/models/obj_01.PLY', help='Path to object mesh file (PLY or OBJ)')
  parser.add_argument('--mesh_scale', type=float, default=0.001, help='Uniform scale factor applied to mesh geometry before inference')
  parser.add_argument('--cam_K_file', type=str, default=f'{code_dir}/dataset/cam_K.txt', help='Optional camera intrinsic matrix file override')
  parser.add_argument('--mask_file', type=str, default=None, help='Optional fixed mask file used for initialization')
  parser.add_argument('--rs_color_width', type=int, default=960)
  parser.add_argument('--rs_color_height', type=int, default=540)
  parser.add_argument('--rs_depth_width', type=int, default=1280)
  parser.add_argument('--rs_depth_height', type=int, default=720)
  parser.add_argument('--rs_fps', type=int, default=30)
  parser.add_argument('--rs_serial', type=str, default=None, help='Optional RealSense device serial number')
  parser.add_argument('--shorter_side', type=int, default=None, help='Resize shorter side to this value')
  parser.add_argument('--zfar', type=float, default=np.inf, help='Depth far clipping plane')
  parser.add_argument('--depth_scale', type=float, default=1.0, help='Multiplier applied to depth values before inference')
  parser.add_argument('--est_refine_iter', type=int, default=1)
  parser.add_argument('--track_refine_iter', type=int, default=1)
  parser.add_argument('--show_depth', type=int, default=0, help='Display colorized depth map alongside inference')
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/output/realsense_stream')
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

  if rs is None:
    raise RuntimeError('pyrealsense2 is not available')
  if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available. Please run on a machine with CUDA drivers and a compatible GPU.')

  torch.cuda.set_device(args.device)
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  torch.backends.cuda.matmul.allow_tf32 = True

  cam_K = load_camera_matrix(args.cam_K_file)
  reader = RealSenseStreamReader(
    color_width=args.rs_color_width,
    color_height=args.rs_color_height,
    depth_width=args.rs_depth_width,
    depth_height=args.rs_depth_height,
    fps=args.rs_fps,
    shorter_side=args.shorter_side,
    zfar=args.zfar,
    serial=args.rs_serial,
    cam_K=cam_K,
    mask_file=args.mask_file,
  )

  mesh = trimesh.load(args.mesh_file)
  verify_mesh_texture(mesh, args.mesh_file)
  
  if args.mesh_scale != 1.0:
    if not hasattr(mesh, 'apply_scale'):
      raise RuntimeError(f'Loaded mesh type does not support scaling: {type(mesh)}')
    mesh.apply_scale(args.mesh_scale)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext(args.device)
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                       scorer=scorer, refiner=refiner, debug_dir=args.debug_dir, debug=args.debug, glctx=glctx)
  est.to_device(f'cuda:{args.device}')

  if args.save_video:
    os.makedirs(args.debug_dir, exist_ok=True)
    writer = make_video_writer(os.path.join(args.debug_dir, 'realsense_stream.mp4'), reader.W, reader.H, args.rs_fps)
  else:
    writer = None

  frame_idx = 0
  frame_skip_count = 0
  frames_to_skip = 30
  mask_consumed = False
  try:
    while True:
      torch.cuda.synchronize()
      frame_start = time.time()
      frame = reader.read()
      if frame is None:
        continue

      # Skip first 30 frames for camera initialization
      if frame_skip_count < frames_to_skip:
        frame_skip_count += 1
        logging.info(f'Skipping frame {frame_skip_count}/{frames_to_skip} for camera initialization')
        continue

      color = frame['color']
      depth = frame['depth']
      mask = frame['mask']
      if args.depth_scale != 1.0:
        depth = depth.astype(np.float32, copy=False) * args.depth_scale

      has_mask = (mask is not None) and np.any(mask)
      # Use provided mask only for the very first registration frame.
      if frame_idx == 0:
        init_mask = mask if has_mask else (depth >= 0.001).astype(np.uint8)
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=init_mask, iteration=args.est_refine_iter)
        if has_mask and getattr(reader, 'fixed_mask', None) is not None:
          reader.fixed_mask = None
      elif est.pose_last is None:
        # Re-registration after tracking loss: do not use external mask, rely on depth
        init_mask = (depth >= 0.001).astype(np.uint8)
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=init_mask, iteration=args.est_refine_iter)
      else:
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

      torch.cuda.synchronize()
      torch.cuda.empty_cache()
      frame_time = time.time() - frame_start
      logging.info(f'Frame {frame_idx + 1} inference time: {frame_time:.3f} sec')

      if args.print_pose_info:
        translation, euler_angles = extract_translation_and_rotation(pose)
        print(f'\n--- Frame {frame_idx} ---')
        print(f'Translation (meters): x={translation[0]:.6f}, y={translation[1]:.6f}, z={translation[2]:.6f}')
        print(f'Distance from camera: {np.linalg.norm(translation):.6f} meters')
        print(f'Rotation (Euler angles in degrees): roll={euler_angles[0]:.2f}°, pitch={euler_angles[1]:.2f}°, yaw={euler_angles[2]:.2f}°')

      os.makedirs(f'{args.debug_dir}/ob_in_cam', exist_ok=True)
      np.savetxt(f'{args.debug_dir}/ob_in_cam/{frame["id_str"]}.txt', pose.reshape(4, 4))

      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)

      if args.debug >= 1:
        cv2.imshow('FoundationPose RealSense stream', vis[..., ::-1])
        if args.show_depth:
          # Colorize depth map for visualization
          depth_normalized = np.clip(depth / np.max(depth[depth > 0]) if np.any(depth > 0) else depth, 0, 1)
          depth_colorized = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
          cv2.imshow('Depth Map (Colorized)', depth_colorized)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
          break

      if args.debug >= 2:
        os.makedirs(f'{args.debug_dir}/track_vis', exist_ok=True)
        imageio.imwrite(f'{args.debug_dir}/track_vis/{frame["id_str"]}.png', vis)

      if writer is not None:
        writer.write(vis[..., ::-1])

      frame_idx += 1
  finally:
    if writer is not None:
      writer.release()
    reader.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
