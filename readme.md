# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
[[Paper]](https://arxiv.org/abs/2312.08344) [[Website]](https://nvlabs.github.io/FoundationPose/)

This is the official implementation of our paper to be appeared in CVPR 2024 (Highlight)

Contributors: Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield

We present FoundationPose, a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups. Our approach can be instantly applied at test-time to a novel object without fine-tuning, as long as its CAD model is given, or a small number of reference images are captured. We bridge the gap between these two setups with a neural implicit representation that allows for effective novel view synthesis, keeping the downstream pose estimation modules invariant under the same unified framework. Strong generalizability is achieved via large-scale synthetic training, aided by a large language model (LLM), a novel transformer-based architecture, and contrastive learning formulation. Extensive evaluation on multiple public datasets involving challenging scenarios and objects indicate our unified approach outperforms existing methods specialized for each task by a large margin. In addition, it even achieves comparable results to instance-level methods despite the reduced assumptions.


<img src="assets/intro.jpg" width="70%">

**🤖 For ROS version, please check [Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation), which enjoys TRT fast inference and C++ speed up.**

## Demos

Robotic Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/aa341004-5a15-4293-b3da-000471fd74ed


AR Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/80e96855-a73c-4bee-bcef-7cba92df55ca


Results on YCB-Video dataset:

https://github.com/NVlabs/FoundationPose/assets/23078192/9b5bedde-755b-44ed-a973-45ec85a10bbe


# Data prepare


1) Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

1) [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `demo_data/`


# Env setup : docker
  ```
  cd docker/
  docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose  # Or to build from scratch: docker build --network host -t foundationpose .
  bash docker/run_container_falku.sh
  ```


If it's the first time you launch the container, you need to build extensions. Run this command *inside* the Docker container.
```
bash build_all_falku.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```

For more recent GPU such as 4090, refer to [this](https://github.com/NVlabs/FoundationPose/issues/27).
In short, do the following:
```
docker pull shingarey/foundationpose_custom_cuda121:latest
```
Then modify the bash script to use this image instead of `foundationpose:latest`.


# Run model-based demo
The paths have been set in argparse by default. If you need to change the scene, you can pass the args accordingly. By running on the demo data, you should be able to see the robot manipulating the mustard bottle. Pose estimation is conducted on the first frame, then it automatically switches to tracking mode for the rest of the video. The resulting visualizations will be saved to the `debug_dir` specified in the argparse. (Note the first time running could be slower due to online compilation)
```
python run_demo.py
```


<img src="assets/demo.jpg" width="50%">


Feel free to try on other objects (**no need to retrain**) such as driller, by changing the paths in argparse.

<img src="assets/demo_driller.jpg" width="50%">


# Run on video files and live streams

This repository includes three custom inference scripts for running FoundationPose on different input sources:

## Single-object video directory inference
**`run_inference.py`** - Runs pose estimation and tracking on a single object using a video directory or video file with optional depth and mask folders.

Setup your data with the following structure:
```
video_dir/
├── rgb/              # RGB frames
├── depth/            # Depth frames
├── masks/            # Optional object masks
└── cam_K.txt         # Camera intrinsic matrix
```

Run with a video directory:
```bash
python run_inference.py \
  --mesh_file /path/to/object.obj \
  --video_dir /path/to/video_dir \
  --debug 1 \
  --debug_dir output/inference
```

Run with a video file:
```bash
python run_inference.py \
  --mesh_file /path/to/object.obj \
  --video_file video.mp4 \
  --depth_dir /path/to/depth_frames \
  --cam_K_file cam_K.txt \
  --save_video 1 \
  --debug_dir output/inference
```

Key arguments:
- `--inference_mode`: Choose `'fast'` (initialize once, then track) or `'frame_registration'` (re-register when mask available). Default: `'fast'`
- `--mesh_scale`: Apply uniform scale to mesh if in different units (e.g., mm instead of m)
- `--print_pose_info`: Print translation and Euler angles to terminal for each frame


## Multi-object video directory inference
**`run_inference_multi_class.py`** - Runs pose estimation for **multiple objects** simultaneously in a single scene. Each object is tracked independently based on mask annotations.

**Important**: Object tracking is based on name matching. Mask folder names must exactly match the model file names (without extension).

Setup your data structure:
```
models_dir/
├── object_1.obj              # Model file for object 1
├── object_2.ply              # Model file for object 2
└── ...

scene_dir/
├── rgb/                      # RGB frames (e.g., 000000.png, 000001.png)
├── depth/                    # Depth frames (same names as RGB)
├── masks/                    # Mask files named with object name
│   ├── object_1.png          # Mask for object_1 at frame 000000
│   ├── object_2.png          # Mask for object_2 at frame 000000
│   └── ...
└── cam_K.txt                 # Camera intrinsic matrix
```

Run multi-object inference:
```bash
python run_inference_multi_class.py \
  --video_dir /path/to/scene_dir \
  --models_dir /path/to/models_dir \
  --mesh_scale 1.0 \
  --debug 1 \
  --show_live 1 \
  --debug_dir output/multi_class
```

Key arguments:
- `--models_dir`: Directory containing 3D model files (.obj, .ply, .stl). **Mask folder names must match the model file stem (e.g., `object_1.obj` → `masks/object_1.png`)**
- `--mesh_scale`: Uniform scale applied to all loaded models
- `--show_live`: Display live OpenCV window during inference
- `--save_video`: Save output visualization as MP4
- `--save_pose_txt`: Save per-frame poses to `debug/ob_in_cam/`
- `--save_results_json`: Save all poses to `debug_dir/poses.json`


## RealSense live stream inference
**`run_realsense_stream.py`** - Runs pose estimation and tracking on a live RGB-D stream from an Intel RealSense camera, tracking a single object in real-time.


Run with RealSense:
```bash
python run_realsense_stream.py \
  --mesh_file /path/to/object.obj \
  --cam_K_file cam_K.txt \
  --est_refine_iter 1 \
  --track_refine_iter 1 \
  --debug 1 \
  --debug_dir output/realsense
```

Key arguments:
- `--mesh_scale`: Scale factor for the mesh
- `--mask_file`: Optional fixed mask image for initialization (e.g., pre-captured object region)
- `--rs_color_width`, `--rs_color_height`: Color stream resolution (default 960×540)
- `--rs_depth_width`, `--rs_depth_height`: Depth stream resolution (default 1280×720)
- `--rs_fps`: Frames per second (default 30)
- `--rs_serial`: Serial number of specific RealSense device (if multiple connected)
- `--show_depth`: Display colorized depth map alongside inference
- `--print_pose_info`: Print pose (translation + rotation) for each frame
- `--save_video`: Save inference output as MP4


# Run on public datasets (LINEMOD, YCB-Video)

For this you first need to download LINEMOD dataset and YCB-Video dataset.

To run model-based version on these two datasets respectively, set the paths based on where you download. The results will be saved to `debug` folder
```
python run_linemod.py --linemod_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD --use_reconstructed_mesh 0

python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 0
```

To run model-free few-shot version. You first need to train Neural Object Field. `ref_view_dir` is based on where you download in the above "Data prepare" section. Set the `dataset` flag to your interested dataset.
```
python bundlesdf/run_nerf.py --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16 --dataset ycbv
```

Then run the similar command as the model-based version with some small modifications. Here we are using YCB-Video as example:
```
python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 1 --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16
```
