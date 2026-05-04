import numpy as np

npz_path = r"C:\Users\kfakh\OneDrive\Desktop\falku\camera_calibration.npz"
output_txt = "K.txt"

# Load .npz file
data = np.load(npz_path)

print("Available keys:", data.files)

# Common intrinsic keys
possible_keys = ["K", "intrinsic", "camera_matrix", "mtx"]

K = None
for key in possible_keys:
    if key in data.files:
        K = data[key]
        print(f"Found intrinsics under key: {key}")
        break

# Fallback if unknown structure
if K is None:
    print("No standard key found, using first array.")
    K = data[data.files[0]]

# Ensure correct shape
K = np.array(K).reshape(3, 3)

# Save to TXT in OpenCV format
with open(output_txt, "w") as f:
    for row in K:
        f.write(" ".join(f"{v:.18e}" for v in row) + "\n")

print(f"\n Intrinsics saved to: {output_txt}")

# Optional print
print("\n Intrinsic Matrix:")
print(K)
