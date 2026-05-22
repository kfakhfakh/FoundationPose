import os
import cv2
import numpy as np

labels_dir =  os.path.expanduser("~/Downloads/hhhh/labels/train")
output_dir = "masks"

os.makedirs(output_dir, exist_ok=True)

# ⚠️ SET YOUR IMAGE SIZE HERE
IMG_WIDTH = 960
IMG_HEIGHT = 540


for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, file)

    # empty mask
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue  # not a segmentation polygon

        coords = list(map(float, parts[1:]))

        pts = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * IMG_WIDTH)
            y = int(coords[i + 1] * IMG_HEIGHT)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)

        cv2.fillPoly(mask, [pts], 255)

    out_name = os.path.splitext(file)[0] + ".png"
    out_path = os.path.join(output_dir, out_name)

    cv2.imwrite(out_path, mask)
    print(f"Saved: {out_path}")

print("✅ Done generating masks without images")
