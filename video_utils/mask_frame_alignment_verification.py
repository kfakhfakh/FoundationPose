import os
import cv2
import numpy as np

input_dir = r"C:\Users\kfakh\OneDrive\Desktop\falku\FoundationPose\video_utils\ref_views_16\ob_0000001\rgb"
mask_dir  = r"C:\Users\kfakh\OneDrive\Desktop\falku\FoundationPose\video_utils\ref_views_16\ob_0000001\mask"

output_dir = os.path.join(os.path.dirname(input_dir), "masked_output")
os.makedirs(output_dir, exist_ok=True)


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


for file in os.listdir(input_dir):
    if not file.lower().endswith((".png", ".jpg", ".tif", ".tiff")):
        continue

    img_path  = os.path.join(input_dir, file)
    mask_path = os.path.join(mask_dir, os.path.splitext(file)[0] + ".png")

    if not os.path.exists(mask_path):
        print(f"Missing mask for {file}")
        continue

    img  = load_image(img_path)
    mask = load_mask(mask_path)

    if img is None or mask is None:
        continue

    # resize mask if needed
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 🔍 detect image type
    if len(img.shape) == 2:
        # ✅ depth or grayscale (H, W)
        masked = img * mask

    elif len(img.shape) == 3:
        # ✅ RGB or multi-channel (H, W, C)
        channels = img.shape[2]

        # expand mask to match channels
        mask_expanded = np.repeat(mask[:, :, None], channels, axis=2)
        masked = img * mask_expanded

    else:
        print(f"Unsupported shape for {file}: {img.shape}")
        continue

    out_path = os.path.join(output_dir, file)
    cv2.imwrite(out_path, masked)

    print(f"Saved: {out_path}")

print("Done")
