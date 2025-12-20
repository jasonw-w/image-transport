import cv2
import numpy as np
import os

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, 'imgs')
    target_path = os.path.join(img_dir, 'target (1).png')
    output_path = os.path.join(img_dir, 'random_noise.png')

    # Load target image to get dimensions
    print(f"Loading image from: {target_path}")
    target_img = cv2.imread(target_path)
    
    if target_img is None:
        print("Error: Could not load target image.")
        return

    height, width, channels = target_img.shape
    print(f"Target dimensions: {width}x{height}, Channels: {channels}")

    # Generate random noise
    # np.random.randint returns integers, so we specify range [0, 256) for uint8
    noise_img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

    # Save the noise image
    cv2.imwrite(output_path, noise_img)
    print(f"Random noise image saved to: {output_path}")

if __name__ == "__main__":
    main()
