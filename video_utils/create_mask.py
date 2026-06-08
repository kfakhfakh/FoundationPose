import cv2
import numpy as np

def polygon_to_mask(image_path, label_path):
    # Read image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Create empty mask
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        
        # first value is class id
        class_id = int(parts[0])
        
        # remaining values are polygon points (normalized)
        polygon = np.array(parts[1:]).reshape(-1, 2)

        # convert normalized coords -> pixel coords
        polygon[:, 0] *= w
        polygon[:, 1] *= h

        polygon = polygon.astype(np.int32)

        # fill polygon in mask
        cv2.fillPoly(mask, [polygon], 255)

    return img, mask


# Example usage
image_path = r"C:\Users\kfakh\Downloads\ycbv\ycbv_test_all\test\000048\rgb\000002.png"
label_path = r"C:\Users\kfakh\Downloads\masks\out5\labels\train\000002.txt"

image, mask = polygon_to_mask(image_path, label_path)

cv2.imwrite("mask5.png", mask)

print("Mask shape:", mask.shape)
