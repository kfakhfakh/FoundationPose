import cv2
import os

def frames_to_video(frame_folder, output_video, fps=30):
    # Get sorted list of image files
    images = sorted([
        img for img in os.listdir(frame_folder)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not images:
        print("No images found in folder.")
        return

    # Read first frame to get video size
    first_frame_path = os.path.join(frame_folder, images[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add frames
    for image in images:
        frame_path = os.path.join(frame_folder, image)
        frame = cv2.imread(frame_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")

# Example usage
frames_to_video("C:\\Users\\kfakh\\OneDrive\\Desktop\\falku\\FoundationPose\\debug\\track_vis", "output.mp4", fps=30)
