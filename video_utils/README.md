# Video Utilities

This folder contains helper scripts for video processing, data capture, and preparation for FoundationPose inference. These tools assist with RGB-D data handling, mask generation, camera calibration, and reference view preparation.

## Video Processing Scripts

### Frame and Video Conversion

- **`vid_to_frames.py`** - Extracts frames from video files at a specified rate
  - Supports RGB and depth videos separately
  - Saves frames as PNG images in organized folders
  - Usage: `python vid_to_frames.py` (edit paths in script for `rgb_video_path` and `depth_video_path`)

- **`frames_to_vid.py`** - Combines a sequence of frames into a video file
  - Usage: `python frames_to_vid.py` (edit `frame_folder` and `output_video` in script)

- **`video_sampler.py`** - Records RGB-D stream from RealSense at controlled framerate
  - Captures both RGB and depth frames synchronized
  - Saves to organized `rgb/` and `depth/` folders
  - Usage: `python video_sampler.py` (configure `BASE_FOLDER` and `SAVE_FPS` in script)

### Depth Processing and Visualization

- **`depth_rgb_viewer.py`** - Interactive side-by-side viewer for RGB and depth frames with dataset support
  - Loads frames from dataset folders (rgb/ and depth/ subfolders)
  - Click to query depth values at pixel locations
  - Supports playback with mouse interactions
  - Supports 16-bit depth images
  - Usage: `python depth_rgb_viewer.py --dataset /path/to/dataset/` or `python depth_rgb_viewer.py --rgb image.png --depth depth.png`

- **`depth_vis.py`** - Visualizes and compares raw vs. enhanced depth maps
  - Normalizes depth values for display with optional colormap (JET)
  - Side-by-side comparison of two depth images
  - Usage: Edit image paths in script and run `python depth_vis.py`

- **`enhanced_depth_gen.py`** - Creates masked depth maps for reference view preparation
  - Applies object masks to depth images, keeping only masked regions
  - Zeroes out depth values outside mask boundaries
  - Outputs to `depth_enhanced/` folder for model-free pipeline
  - Usage: Edit `depth_dir` and `mask_dir` paths in script and run `python enhanced_depth_gen.py`

### Mask Generation

- **`mask_generator.py`** - Interactive RealSense-based mask generator with adjustable rectangle
  - Live stream from RealSense camera with trackbars to adjust mask position and size
  - Supports saving masks to file
  - Usage: `python mask_generator.py --mode interactive` (requires RealSense connected)

- **`seg_mask_gen.py`** - Generates binary masks from YOLO polygon segmentation annotations
  - Converts normalized polygon coordinates from YOLO format to binary mask images
  - Usage: Edit `labels_dir`, `IMG_WIDTH`, and `IMG_HEIGHT` in script and run `python seg_mask_gen.py`

- **`create_mask.py`** - Creates binary masks from YOLO polygon annotations
  - Converts normalized polygon points to pixel coordinates and fills polygon regions
  - Usage: Edit `image_path` and `label_path` in script and run `python create_mask.py`

### Data Verification and Application

- **`mask_frame_alignment_verification.py`** - Applies masks to images and saves masked output
  - Two modes: `dir` (batch process images) and `realsense` (live stream)
  - In dir mode: loads images from folder and applies corresponding masks
  - In realsense mode: captures live RealSense frames and applies mask(s)
  - Supports single fixed mask or sequential mask frames
  - Usage (dir): `python mask_frame_alignment_verification.py --mode dir --input_dir rgb/ --mask_dir masks/ --output_dir masked_output/`
  - Usage (RealSense): `python mask_frame_alignment_verification.py --mode realsense --mask_file mask.png --output_dir masked_stream/`

- **`snapshot.py`** - RealSense camera viewer with snapshot capture
  - Displays live RGB and depth streams
  - Detects chessboard patterns for calibration verification
  - Press SPACE to capture snapshot, ESC to exit
  - Optional: loads and applies calibration if available
  - Usage: `python snapshot.py` (optional: update `CALIB_FILE` path for calibration)

### Camera Calibration and Coordinate Systems

- **`K_gen.py`** - Extracts camera intrinsic matrix from .npz calibration file
  - Loads intrinsics from numpy file and saves as K.txt (3×3 matrix format)
  - Supports various key names (K, intrinsic, camera_matrix, mtx)
  - Usage: Edit `npz_path` in script and run `python K_gen.py`

- **`plot_cam_in_obj_axis.py`** - Projects coordinate axes onto RGB images
  - Visualizes camera pose relative to object frame using cam_in_ob matrices
  - Draws X (red), Y (green), Z (blue) axes
  - Processes dataset with rgb/, cam_in_ob/, and K.txt folders
  - Usage: `python plot_cam_in_obj_axis.py --root /path/to/ref_views_*/ob_*/ --output /path/to/output/`

- **`realsense_test.py`** - Tests RealSense camera and displays live streams
  - Shows synchronized RGB and depth from RealSense camera
  - Displays center pixel depth value in real-time
  - Detects chessboard patterns for calibration verification
  - Usage: `python realsense_test.py`

### Data Capture

- **`aruco_obj_rgb_depth_capture.py`** - Captures RGB-D data with ArUco marker-based pose tracking
  - Records synchronized RGB-D frames from RealSense camera
  - Detects ArUco markers in frames and estimates their 3D pose
  - Allows interactive transform adjustment with keyboard controls (a/z for X, e/r for Y, q/s for Z, etc.)
  - Saves frames with corresponding camera-to-object (cam_in_ob) pose matrices
  - Usage: Run `python aruco_obj_rgb_depth_capture.py` (configure marker ID and size in script)

## Reference Views Folders

### `ref_views_10/` 

These folders contain preprocessed reference views used for the model-free/few-shot FoundationPose workflow.

Common structure:
- **`ob_0000001/`**
  - `rgb/` - Reference RGB images
  - `depth/` - Reference depth maps
  - `depth_enhanced/` - Masked/enhanced depth maps
  - `mask/` - Object masks
  - `model/` - Object model assets
  - `K.txt` - Camera intrinsics
  - `select_frames.yml` - Selected reference frame list
  - `cam_in_ob/` - Camera poses in object frame


## Typical Workflow (Model-Free Setup)

1. **Calibrate camera intrinsics**
  - See [calibrate/README.md](../calibrate/README.md) for detailed camera calibration procedures.
  - Extract and save K.txt intrinsic matrix and calibration.npz files to be used throughout the pipeline.

2. **Capture object RGB-D sequence**
  - Use `video_sampler.py` for simple capture, or
  - use `aruco_obj_rgb_depth_capture.py` if you also need `cam_in_ob` pose files (requires camera calibration from step 1).

3. **Build frame folders**
  - Ensure you have frame-aligned `rgb/` and `depth/`.
  - If input is video files, extract frames with `vid_to_frames.py`.

4. **Check RGB/depth quality and alignment**
  - Use `depth_rgb_viewer.py`.

5. **Annotate object masks (external step)**
  - Create polygon segmentation labels (YOLO format) using your annotation tool.

6. **Convert labels to binary masks**
  - Use `create_mask.py` or `seg_mask_gen.py` to generate mask images.

7. **Create masked depth maps**
  - Run `enhanced_depth_gen.py` to generate `depth_enhanced/` from `depth/` + masks.

8. **Assemble reference-view structure**
  - Place files under `ref_views_*/ob_xxxxxxx/`:
    - `rgb/`, `depth/`, `depth_enhanced/`, `mask/`, `cam_in_ob/`, `K.txt`, `select_frames.yml`, `model/`.

9. **Validate poses/masks (optional)**
  - Use `plot_cam_in_obj_axis.py` for pose visualization.
  - Use `mask_frame_alignment_verification.py` for masked-image checks.

10. **Run model-free reconstruction/training**
  - From project root, run `bundlesdf/run_nerf.py` with your `--ref_view_dir`.

11. **Run model-free FoundationPose inference**
  - Use the main inference scripts with model-free settings (for example reconstructed model usage).

## Configuration Notes

Most scripts use **hardcoded paths or script-level configuration** (not command-line args):
- Edit file paths directly in the scripts before running
- Look for variables like `BASE_FOLDER`, `depth_dir`, `mask_dir`, `output_dir` at the top of each script
- Some scripts require setting image dimensions: `IMG_WIDTH`, `IMG_HEIGHT`

