import cv2
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_frames(video_path, output_folder, frames_per_sec=4):
    if video_path is None or not os.path.exists(video_path):
        logging.warning(f"Video not found or not provided: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.warning(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.warning(f"Invalid FPS in video: {video_path}")
        return

    frame_interval = max(1, int(fps / frames_per_sec))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_folder, video_name)
    os.makedirs(save_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(save_dir, f"frame_{saved_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    logging.info(f"Saved {saved_count} frames from {video_path}")


def process_videos(rgb_video_path=None, depth_video_path=None, output_base="output", frames_per_sec=4):
    rgb_output = os.path.join(output_base, "rgb")
    depth_output = os.path.join(output_base, "depth")

    os.makedirs(rgb_output, exist_ok=True)
    os.makedirs(depth_output, exist_ok=True)

    # Process RGB video
    if rgb_video_path:
        extract_frames(rgb_video_path, rgb_output, frames_per_sec)
    else:
        logging.warning("RGB video path not provided")

    # Process Depth video
    if depth_video_path:
        extract_frames(depth_video_path, depth_output, frames_per_sec)
    else:
        logging.warning("Depth video path not provided")


if __name__ == "__main__":
    rgb_video = "path/to/rgb_video.mp4"       # <-- set or None
    depth_video = "path/to/depth_video.mp4"   # <-- set or None

    process_videos(
        rgb_video_path=rgb_video,
        depth_video_path=depth_video,
        output_base="output_folder",
        frames_per_sec=4
    )
