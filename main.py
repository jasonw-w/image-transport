def main(src_fit_target,
         brighness_weight, frequency_weight, distance_weight,
         stiffness, damping, stretch_factor,
         source_path, target_path,
         cell_percentage,
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
    if src_fit_target:
        src_img = image_ops.match_aspect_ratio(tgt_img, src_img)
        CELL_SIZE = (src_img.shape[0] * cell_percentage + src_img.shape[1] * cell_percentage) // 2
    else:
        tgt_img = image_ops.match_aspect_ratio(src_img, tgt_img)
        CELL_SIZE = (tgt_img.shape[0] * cell_percentage + tgt_img.shape[1] * cell_percentage) // 2
    print(f"Cell Size: {CELL_SIZE}")
    # --- 2. GRID SPLITTING ---
    N = int(src_img.shape[0] // CELL_SIZE)
    M = int(src_img.shape[1] // CELL_SIZE)
    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB))
    plt.show()
    if N < 1 or M < 1:
        raise ValueError("Cell size is too large. Cell size should be less than or equal to the smaller dimension of the source image.")
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
        duration = 20
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
        src_fit_target=False,
        brighness_weight=1.0,
        frequency_weight=1.0,
        distance_weight=0.001,
        stiffness=0.01, #how quick each cells moves
        damping=0.98, #how quick each cells slows down
        stretch_factor=0.01, #how much each cell is stretched when accelerated, higher = more stretch
        source_path="demo/imgs/random_noise.png",
        target_path="demo/imgs/rem.png",
        cell_percentage=0.01, # Size of each cell in pixels. LOWER Percentage = HIGHER quality (more cells), but slower.
        quick_gen=False # Whether to use quick generation or not. Quick generation is faster but lower quality
    )