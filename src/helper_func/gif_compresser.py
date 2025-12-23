from PIL import Image, ImageSequence

def compress_gif_resize(input_path, output_path, scale=0.9):
    """
    Compresses a GIF by resizing each frame.
    
    Args:
        input_path (str): Path to input GIF
        output_path (str): Path to save compressed GIF
        scale (float): Scale factor (0.5 = 50% original size)
    """
    try:
        with Image.open(input_path) as im:
            frames = []
            durations = [] # Store duration for each frame

            # Print original info
            width, height = im.size
            print(f"Original Size: {width}x{height}, Frames: {im.n_frames}")
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            print(f"New Target Size: {new_width}x{new_height}")

            for frame in ImageSequence.Iterator(im):
                # Resize using LANZOS for high quality downscaling
                new_frame = frame.copy().resize((new_width, new_height), Image.Resampling.LANCZOS)
                frames.append(new_frame)
                
                # Grab duration of this specific frame (default to 100ms if missing)
                durations.append(frame.info.get('duration', 100))
            
            if not frames:
                print("Error: No frames processed!")
                return
            
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=durations, # Pass the list of durations
                loop=0
            )
            print(f"Saved resized GIF to {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\User\Documents\Project\obamify\simulation.gif"
    OUTPUT_FILE = "random2rem.gif"
    
    # Scale 0.7 means 70% of original dimensions (approx 50% file size reduction)
    compress_gif_resize(INPUT_FILE, OUTPUT_FILE, scale=0.49)
