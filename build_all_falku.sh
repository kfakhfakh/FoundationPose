unset LD_LIBRARY_PATH

apt update && apt install -y \
    build-essential \
    make \
    gcc-9 \
    g++-9

# Compatibility symlinks expected by FoundationPose/CMake
ln -s /usr/bin/gcc-9 /usr/bin/gcc-11 || true && \
    ln -s /usr/bin/g++-9 /usr/bin/g++-11 || true && \
    ln -s /usr/bin/make /usr/bin/gmake || true
bash build_all.sh

# 1. Install prerequisites
apt-get update && apt-get install -y curl gpg software-properties-common

# 2. Add the Intel RealSense repo
echo "deb [arch=amd64] https://librealsense.intel.com/Debian/apt-repo focal main" \
  > /etc/apt/sources.list.d/realsense.list

# 3. Import the GPG key by ID
apt-key adv --keyserver keyserver.ubuntu.com --recv-key FB0B24895113F120

# 4. Update apt
apt-get update

# 5. Install the SDK
apt-get install -y librealsense2-utils librealsense2-dev libusb-1.0-0

# 6. Install Python bindings
pip3 install pyrealsense2

apt install -y usbutils

