import os
import cv2
import numpy as np

depth_dir = r"C:\Users\kfakh\OneDrive\Desktop\falku\FoundationPose\video_utils\ref_views_1\ob_0000001\depth"
mask_dir = r"C:\Users\kfakh\OneDrive\Desktop\falku\FoundationPose\video_utils\ref_views_1\ob_0000001\masks"

# output folder created next to depth folder
output_dir = os.path.join(os.path.dirname(depth_dir), "depth_enhanced")
os.makedirs(output_dir, exist_ok=True)


def load_depth(path):
    # supports 16-bit or 8-bit depth images
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return depth


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    # ensure binary mask
    mask = (mask > 0).astype(np.uint8)
    return mask


for file in os.listdir(depth_dir):
    if not file.lower().endswith((".png", ".jpg", ".tif", ".tiff")):
        continue

    depth_path = os.path.join(depth_dir, file)
    mask_path = os.path.join(mask_dir, os.path.splitext(file)[0] + ".png")

    if not os.path.exists(mask_path):
        print(f"Missing mask for {file}")
        continue

    depth = load_depth(depth_path)
    mask = load_mask(mask_path)

    if depth is None or mask is None:
        continue

    # resize mask if needed (safety)
    if depth.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    # apply mask
    enhanced_depth = depth * mask

    # save result
    out_path = os.path.join(output_dir, file)
    cv2.imwrite(out_path, enhanced_depth)

    print(f"Saved: {out_path}")

print("✅ Done generating enhanced depth maps")
