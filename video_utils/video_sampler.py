import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# =========================
# CONFIG
# =========================
BASE_FOLDER = "dataset"   # change this
SAVE_FPS = 6              # frames per second to save
# =========================

rgb_folder = os.path.join(BASE_FOLDER, "rgb")
depth_folder = os.path.join(BASE_FOLDER, "depth")

os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)

# Create pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Timing control (for saving FPS)
save_interval = 1.0 / SAVE_FPS
last_save_time = 0
frame_id = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())  # uint16

        # Visualization only (for display)
        depth_display = cv2.convertScaleAbs(depth_image, alpha=0.03)

        # =========================
        # SAVE FRAMES (controlled FPS)
        # =========================
        current_time = time.time()
        if current_time - last_save_time >= save_interval:

            rgb_path = os.path.join(rgb_folder, f"{frame_id:06d}.png")
            depth_path = os.path.join(depth_folder, f"{frame_id:06d}.png")

            cv2.imwrite(rgb_path, color_image)
            cv2.imwrite(depth_path, depth_image)

            print(f"Saved frame {frame_id}")

            frame_id += 1
            last_save_time = current_time

        # =========================

        # Show (no overlays)
        cv2.imshow("RGB (aligned)", color_image)
        cv2.imshow("Depth (visual)", depth_display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
