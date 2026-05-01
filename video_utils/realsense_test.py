import pyrealsense2 as rs
import numpy as np
import cv2

# Create pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline
profile = pipeline.start(config)

# Create alignment object (align depth to color)
align = rs.align(rs.stream.color)

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()

        # Align depth frame to RGB frame
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # RAW depth (no colorizing)
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize depth for visualization (important!)
        depth_display = cv2.convertScaleAbs(depth_image, alpha=0.03)

        # Center pixel
        h, w, _ = color_image.shape
        x = w // 2
        y = h // 2

        # Get depth at that pixel (meters)
        dist = depth_frame.get_distance(x, y)

        # Draw marker on RGB
        cv2.circle(color_image, (x, y), 8, (0, 0, 255), 2)
        cv2.putText(color_image,
                    f"{dist:.2f} m",
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)

        # Draw marker on depth
        cv2.circle(depth_display, (x, y), 8, (255, 255, 255), 2)

        # Show images
        cv2.imshow("RGB (aligned)", color_image)
        cv2.imshow("Depth (raw)", depth_display)

        print("Distance at center pixel:", dist, "meters")

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
