def main(brighness_weight, frequency_weight, distance_weight,
         stiffness, damping, stretch_factor,
         source_path, target_path,
         cell_size,
         quick_gen):
    import cv2
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # New Modules
    from src import image_ops
    from src import features
    from src import matcher
    from src import visualizer

    # --- CONFIGURATION ---
    SOURCE_PATH = source_path
    TARGET_PATH = target_path
    CELL_SIZE = cell_size

    # Export Settings
    EXPORT_MP4 = True  # Set to True to save animation
    OUTPUT_FILENAME = "simulation.gif"

    # --- 1. LOAD & PREPARE ---
    src_img = cv2.imread(SOURCE_PATH) 
    tgt_img = cv2.imread(TARGET_PATH)

    #cv2.imshow("Source", src_img)
    #cv2.imshow("Target", tgt_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Resize source to match target aspect/dims
    src_img = image_ops.match_aspect_ratio(tgt_img, src_img)

    # --- 2. GRID SPLITTING ---
    N = src_img.shape[0] // CELL_SIZE
    M = src_img.shape[1] // CELL_SIZE

    print(f"Grid Size: {N}x{M} = {N*M} cells")
    if N*M > 50000:
        print("WARNING: High cell count detected. This might take a while or crash memory.")

    src_cells = image_ops.split(N, M, src_img)
    tgt_cells = image_ops.split(N, M, tgt_img)

    # --- 3. FEATURE EXTRACTION ---
    print("Extracting features (Brightness, Frequency)...")
    src_brightness = features.compute_brightness(src_cells)
    tgt_brightness = features.compute_brightness(tgt_cells)

    src_frequency = features.compute_frequency(src_cells)
    tgt_frequency = features.compute_frequency(tgt_cells)

    # --- 4. MATCHING (The Heavy Part) ---
    # We calculate the distance matrix...
    print("Computing distance matrix...")
    dist_mat = matcher.compute_distance_matrix(N, M) 

    CACHE_FILE = "matches_cache.npy"

    if os.path.exists(CACHE_FILE):
        print(f"Loading matches from {CACHE_FILE}...")
        try:
            matches = np.load(CACHE_FILE)
            # Simple check if cache matches current grid size
            if len(matches) != len(tgt_cells):
                print("Cache size mismatch! Re-computing...")
                raise ValueError("Cache Invalid")
        except Exception:
            matches = matcher.compute_matches(
                src_brightness, src_frequency, 
                tgt_brightness, tgt_frequency, 
                dist_mat,
                alpha=brighness_weight,
                beta=frequency_weight,
                gamma=distance_weight)
            np.save(CACHE_FILE, matches)
    else:
        matches = matcher.compute_matches(
            src_brightness, src_frequency, 
            tgt_brightness, tgt_frequency, 
            dist_mat,
            alpha=brighness_weight,
            beta=frequency_weight,
            gamma=distance_weight)
        np.save(CACHE_FILE, matches)
        print(f"Saved matches to {CACHE_FILE}")

    # --- 5. RENDER ---
    # Optional: Static reconstruction
    # mysaic = image_ops.reconstruct_mosaic(matches, src_cells, src_img.shape, N, M) 
    if quick_gen:
        fps = 15
        render_scale = 0.5
        hold_duration = 0.5
        stretch_factor = 0.2
        duration = 10
    else:
        fps = 30
        duration = 20
        render_scale = 1.0
        hold_duration = 0.25
    # Animation
    output_file = visualizer.render_obamify_voronoi(
        matches, src_cells, src_img.shape, N, M, 
        fps=fps, 
        duration=duration, 
        render_scale=render_scale, 
        hold_duration=hold_duration, 
        stiffness=stiffness, 
        damping=damping,
        stretch_factor=stretch_factor,
        bg_img=src_img,
        save_output=OUTPUT_FILENAME if EXPORT_MP4 else None)

if __name__ == "__main__":
    main(
        brighness_weight=1.0,
        frequency_weight=1.0,
        distance_weight=0.01,
        stiffness=0.01, #how quick each cells moves
        damping=0.98, #how quick each cells slows down
        stretch_factor=0.1, #how much each cell is stretched when accelerated, higher = more stretch
        source_path="demo/imgs/random_noise.png",
        target_path="demo/imgs/kirk.png",
        cell_size=1, # Number of cells in the grid, higher cell_size = higher quality but slower
        quick_gen=False # Whether to use quick generation or not. Quick generation is faster but lower quality
    )