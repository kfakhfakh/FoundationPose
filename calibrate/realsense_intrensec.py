import pyrealsense2 as rs

# Create pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams (color + depth optional)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start pipeline
profile = pipeline.start(config)

try:
    # Get stream profiles
    color_profile = profile.get_stream(rs.stream.color)
    depth_profile = profile.get_stream(rs.stream.depth)

    # Convert to video stream profile
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    print("\n=== COLOR INTRINSICS ===")
    print(f"Width:  {color_intrinsics.width}")
    print(f"Height: {color_intrinsics.height}")
    print(f"fx: {color_intrinsics.fx}")
    print(f"fy: {color_intrinsics.fy}")
    print(f"cx: {color_intrinsics.ppx}")
    print(f"cy: {color_intrinsics.ppy}")
    print(f"Distortion model: {color_intrinsics.model}")
    print(f"Distortion coeffs: {color_intrinsics.coeffs}")

    print("\n=== DEPTH INTRINSICS ===")
    print(f"Width:  {depth_intrinsics.width}")
    print(f"Height: {depth_intrinsics.height}")
    print(f"fx: {depth_intrinsics.fx}")
    print(f"fy: {depth_intrinsics.fy}")
    print(f"cx: {depth_intrinsics.ppx}")
    print(f"cy: {depth_intrinsics.ppy}")
    print(f"Distortion model: {depth_intrinsics.model}")
    print(f"Distortion coeffs: {depth_intrinsics.coeffs}")

    # Build camera matrix K for color
    K = [
        [color_intrinsics.fx, 0, color_intrinsics.ppx],
        [0, color_intrinsics.fy, color_intrinsics.ppy],
        [0, 0, 1]
    ]

    print("\n=== COLOR CAMERA MATRIX (K) ===")
    for row in K:
        print(row)

finally:
    pipeline.stop()
