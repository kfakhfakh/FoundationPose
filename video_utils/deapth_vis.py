import cv2
import numpy as np

# Load your depth images (keep original depth values)
depth_raw = cv2.imread(r"C:\Users\kfakh\Downloads\ycbv\ref_views_8\ob_0000021\depth\0000000.png", cv2.IMREAD_UNCHANGED)
depth_enhanced = cv2.imread(r"C:\Users\kfakh\Downloads\ycbv\ref_views_8\ob_0000021\depth_enhanced\0000000.png", cv2.IMREAD_UNCHANGED)

if depth_raw is None or depth_enhanced is None:
    print("Error loading images")
    exit()

# Normalize depth to 0–255 for display
def visualize_depth(depth):
    # Ignore zero values (no depth)
    depth = depth.astype(np.float32)

    # Optional: mask out invalid pixels
    depth[depth == 0] = np.nan

    # Normalize based on min/max (ignoring NaNs)
    min_val = np.nanmin(depth)
    max_val = np.nanmax(depth)

    norm = (depth - min_val) / (max_val - min_val)
    norm = (norm * 255).astype(np.uint8)

    # Replace NaNs back to 0
    norm = np.nan_to_num(norm).astype(np.uint8)

    return norm

vis_raw = visualize_depth(depth_raw)
vis_enhanced = visualize_depth(depth_enhanced)

# Optional: apply colormap (easier to see depth differences)
vis_raw_color = cv2.applyColorMap(vis_raw, cv2.COLORMAP_JET)
vis_enhanced_color = cv2.applyColorMap(vis_enhanced, cv2.COLORMAP_JET)

# Show results
cv2.imshow("Raw Depth (grayscale)", vis_raw)
cv2.imshow("Raw Depth (color)", vis_raw_color)

cv2.imshow("Enhanced Depth (grayscale)", vis_enhanced)
cv2.imshow("Enhanced Depth (color)", vis_enhanced_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
