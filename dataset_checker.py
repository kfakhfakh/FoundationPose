import os

base_folder = r"C:\Users\kfakh\Downloads\ycbv\ycbv_test_all\test\000048"

rgb_folder = os.path.join(base_folder, "rgb")
depth_folder = os.path.join(base_folder, "depth")

rgb_files = set(os.listdir(rgb_folder))
depth_files = set(os.listdir(depth_folder))

# Find mismatches
rgb_only = rgb_files - depth_files
depth_only = depth_files - rgb_files

# Remove RGB files without depth
for f in rgb_only:
    path = os.path.join(rgb_folder, f)
    os.remove(path)
    print("Removed RGB:", path)

# Remove depth files without RGB
for f in depth_only:
    path = os.path.join(depth_folder, f)
    os.remove(path)
    print("Removed Depth:", path)

print("✔ Cleanup complete")
