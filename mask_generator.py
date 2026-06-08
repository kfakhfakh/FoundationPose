import cv2
import numpy as np
import argparse
from typing import Tuple

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class AdjustableRectangleMask:
    """Create adjustable binary rectangle masks in the center of frames."""
    
    def __init__(self, frame_height: int, frame_width: int, 
                 rect_height: int, rect_width: int,
                 center_x: int = None, center_y: int = None):
        """
        Initialize the mask generator.
        
        Args:
            frame_height: Height of the frame
            frame_width: Width of the frame
            rect_height: Height of the rectangle
            rect_width: Width of the rectangle
            center_x: X coordinate of center (default: frame center)
            center_y: Y coordinate of center (default: frame center)
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.rect_height = rect_height
        self.rect_width = rect_width
        
        # Default to frame center if not specified
        self.center_x = center_x if center_x is not None else frame_width // 2
        self.center_y = center_y if center_y is not None else frame_height // 2
    
    def generate_mask(self) -> np.ndarray:
        """Generate binary rectangle mask."""
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        # Calculate rectangle boundaries
        x1 = max(0, self.center_x - self.rect_width // 2)
        x2 = min(self.frame_width, self.center_x + self.rect_width // 2)
        y1 = max(0, self.center_y - self.rect_height // 2)
        y2 = min(self.frame_height, self.center_y + self.rect_height // 2)
        
        # Draw white rectangle
        mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def apply_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply mask to image."""
        mask = self.generate_mask()
        return cv2.bitwise_and(image, image, mask=mask)
    
    def update_rect_size(self, height: int, width: int):
        """Update rectangle dimensions."""
        self.rect_height = max(1, min(height, self.frame_height))
        self.rect_width = max(1, min(width, self.frame_width))
    
    def update_center(self, x: int, y: int):
        """Update rectangle center position."""
        self.center_x = max(self.rect_width // 2, 
                           min(x, self.frame_width - self.rect_width // 2))
        self.center_y = max(self.rect_height // 2, 
                           min(y, self.frame_height - self.rect_height // 2))


def create_mask(frame_height: int, frame_width: int, 
                rect_height: int, rect_width: int) -> np.ndarray:
    """Simple function to create a binary rectangle mask."""
    mask_gen = AdjustableRectangleMask(frame_height, frame_width, 
                                       rect_height, rect_width)
    return mask_gen.generate_mask()


def interactive_demo():
    """Interactive demo with trackbars for adjustment using RealSense camera."""
    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed. Please install it to use RealSense features.")
    
    # Setup RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color and depth streams
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    # Start pipeline
    profile = pipeline.start(config)
    
    # Create alignment object
    align = rs.align(rs.stream.color)
    
    # Get actual frame dimensions
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    frame_width = intr.width
    frame_height = intr.height
    
    # Initialize mask generator with RealSense dimensions
    mask_gen = AdjustableRectangleMask(
        frame_height=frame_height,
        frame_width=frame_width,
        rect_height=frame_height // 2,
        rect_width=frame_width // 2
    )
    
    cv2.namedWindow('Mask Controls')
    
    # Create trackbars for adjustment
    cv2.createTrackbar('Rect Width', 'Mask Controls', frame_width // 2, 
                      frame_width, lambda x: mask_gen.update_rect_size(
                          mask_gen.rect_height, x))
    cv2.createTrackbar('Rect Height', 'Mask Controls', frame_height // 2, 
                      frame_height, lambda x: mask_gen.update_rect_size(
                          x, mask_gen.rect_width))
    cv2.createTrackbar('Center X', 'Mask Controls', frame_width // 2, 
                      frame_width, lambda x: mask_gen.update_center(
                          x, mask_gen.center_y))
    cv2.createTrackbar('Center Y', 'Mask Controls', frame_height // 2, 
                      frame_height, lambda x: mask_gen.update_center(
                          mask_gen.center_x, x))
    
    print("RealSense Controls:")
    print(f"- Camera resolution: {frame_width}×{frame_height}")
    print("- Adjust trackbars to change rectangle position and size")
    print("- Press 'q' to quit")
    print("- Press 's' to save current mask")
    
    frame_count = 0
    try:
        while True:
            # Get frames from RealSense
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            # RealSense provides BGR, keep it as-is for cv2.imshow (which expects BGR)
            
            # Generate mask
            mask = mask_gen.generate_mask()
            
            # Create visualization
            masked_image = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Draw rectangle outline on original frame for reference
            x1 = mask_gen.center_x - mask_gen.rect_width // 2
            x2 = mask_gen.center_x + mask_gen.rect_width // 2
            y1 = mask_gen.center_y - mask_gen.rect_height // 2
            y2 = mask_gen.center_y + mask_gen.rect_height // 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display side by side
            combined = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            
            cv2.imshow('Mask Controls', combined)
            cv2.imshow('Masked Image', masked_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'mask_{frame_count:04d}.png', mask)
                print(f"✓ Saved mask_{frame_count:04d}.png ({frame_width}×{frame_height})")
                frame_count += 1
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("RealSense Mask Generator for Adjustable Rectangle Masks")
    parser = argparse.ArgumentParser(description='Generate adjustable rectangle masks for RealSense')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['interactive', 'create', 'auto-detect'],
                       help='Mode: interactive (RealSense live), create (static mask), auto-detect (detect from camera)')
    parser.add_argument('--frame-height', type=int, default=None,
                       help='Frame height (default: auto-detect from RealSense)')
    parser.add_argument('--frame-width', type=int, default=None,
                       help='Frame width (default: auto-detect from RealSense)')
    parser.add_argument('--rect-height', type=int, default=None,
                       help='Rectangle height (default: half frame height)')
    parser.add_argument('--rect-width', type=int, default=None,
                       help='Rectangle width (default: half frame width)')
    parser.add_argument('--output', type=str, default='realsense_mask.png',
                       help='Output mask filename')
    parser.add_argument('--center-x', type=int, default=None,
                       help='Center X coordinate (default: frame center)')
    parser.add_argument('--center-y', type=int, default=None,
                       help='Center Y coordinate (default: frame center)')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_demo()
    elif args.mode == 'auto-detect':
        # Auto-detect RealSense dimensions
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed. Please install it to use RealSense features.")
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        frame_width = intr.width
        frame_height = intr.height
        
        pipeline.stop()
        
        rect_width = args.rect_width if args.rect_width is not None else frame_width // 2
        rect_height = args.rect_height if args.rect_height is not None else frame_height // 2
        center_x = args.center_x if args.center_x is not None else frame_width // 2
        center_y = args.center_y if args.center_y is not None else frame_height // 2
        
        mask_gen = AdjustableRectangleMask(frame_height, frame_width, rect_height, rect_width, center_x, center_y)
        mask = mask_gen.generate_mask()
        
        success = cv2.imwrite(args.output, mask)
        if success:
            saved_mask = cv2.imread(args.output, cv2.IMREAD_GRAYSCALE)
            print(f"✓ Mask saved to {args.output}")
            print(f"  Frame dimensions (auto-detected from RealSense): {frame_width}×{frame_height} (W×H)")
            print(f"  Saved mask dimensions: {saved_mask.shape[1]}×{saved_mask.shape[0]} (W×H)")
            print(f"  Rectangle: {rect_width}×{rect_height}")
        else:
            print(f"ERROR: Failed to save mask to {args.output}")
    else:
        # Create static mask with provided dimensions
        frame_width = args.frame_width if args.frame_width is not None else 960
        frame_height = args.frame_height if args.frame_height is not None else 540
        rect_width = args.rect_width if args.rect_width is not None else frame_width // 2
        rect_height = args.rect_height if args.rect_height is not None else frame_height // 2
        center_x = args.center_x if args.center_x is not None else frame_width // 2
        center_y = args.center_y if args.center_y is not None else frame_height // 2
        
        mask_gen = AdjustableRectangleMask(frame_height, frame_width, rect_height, rect_width, center_x, center_y)
        mask = mask_gen.generate_mask()
        
        success = cv2.imwrite(args.output, mask)
        if success:
            saved_mask = cv2.imread(args.output, cv2.IMREAD_GRAYSCALE)
            print(f"✓ Mask saved to {args.output}")
            print(f"  Frame dimensions: {frame_width}×{frame_height} (W×H)")
            print(f"  Saved mask dimensions: {saved_mask.shape[1]}×{saved_mask.shape[0]} (W×H)")
            print(f"  Rectangle: {rect_width}×{rect_height}")
        else:
            print(f"ERROR: Failed to save mask to {args.output}")
