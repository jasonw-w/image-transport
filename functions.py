import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import EllipseCollection
import time
def match_aspect_ratio(source_img, target_img):
    """
    Crops the target_img to match the aspect ratio of source_img,
    then resizes target_img to match source_img dimensions exactly.
    """
    h_s, w_s = source_img.shape[:2]
    h_t, w_t = target_img.shape[:2]

    source_aspect = w_s / h_s
    target_aspect = w_t / h_t

    # 1. Crop Target to match Source Aspect Ratio
    if target_aspect > source_aspect:
        # Target is too wide: Crop the sides
        new_w = int(h_t * source_aspect)
        start_x = (w_t - new_w) // 2
        cropped_target = target_img[:, start_x : start_x + new_w]
    else:
        # Target is too tall: Crop the top/bottom
        new_h = int(w_t / source_aspect)
        start_y = (h_t - new_h) // 2
        cropped_target = target_img[start_y : start_y + new_h, :]

    # 2. Resize to exact Source dimensions (Optional but safer)
    final_target = cv2.resize(cropped_target, (w_s, h_s))

    return final_target

def split(N, M, img):
    """
    Splits an image into N rows and M columns.
    Automatically resizes the image if dimensions aren't perfectly divisible.
    """
    H, W = img.shape[0], img.shape[1]

    # 1. Calculate the perfect new dimensions
    # We strip off the remainder to make it perfectly divisible
    # e.g., if W=1005 and M=10, remainder is 5. new_W = 1000.
    new_H = H - (H % N)
    new_W = W - (W % M)

    # 2. Check if resize is needed
    if H != new_H or W != new_W:
        print(f"Resizing image from ({W}x{H}) to ({new_W}x{new_H}) to fit grid.")
        # cv2.resize expects (width, height)
        img = cv2.resize(img, (new_W, new_H))

    # 3. Define Cell Dimensions (Now guaranteed to be integers)
    cell_h = new_H // N
    cell_w = new_W // M

    cells = []
    for i in range(N):
        for j in range(M):
            y_start = i * cell_h
            y_end   = (i + 1) * cell_h
            x_start = j * cell_w
            x_end   = (j + 1) * cell_w

            cell = img[y_start:y_end, x_start:x_end, :]
            cells.append(cell)

    return cells

def normalize_stats(data):
    """
    Normalizes data to a range of [0, 1].
    """
    data = np.array(data)

    # Case 1: 1D array (Just brightness)
    if data.ndim == 1:
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val

        # Avoid division by zero
        if range_val == 0:
            range_val = 1

        return (data - min_val) / range_val

    # Case 2: 2D array (Brightness + Frequency)
    else:
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        range_vals = maxs - mins

        # Avoid division by zero for each column
        range_vals[range_vals == 0] = 1

        return (data - mins) / range_vals

def compute_brightness(cells):
    """
    Computes the brightness of each cell.
    """
    averages = []
    for cell in cells:
        sum = 0
        for row in cell:
            for pixel in row:
                sum += 0.299*pixel[0] + 0.589*pixel[1] + 0.114*pixel[2]
        averages.append(sum/(len(cell)*len(row)))
    return normalize_stats(np.array(averages))

def compute_frequency(cells):
    """
    Computes the frequency of each cell.
    """
    freqs = []
    for cell in cells:
        grey = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(grey, cv2.CV_64F)
        edge_magnitude = np.abs(laplacian)
        frequency_score = np.mean(edge_magnitude)
        freqs.append(frequency_score)
    return(normalize_stats(freqs))

def compute_distance_matrix(N, M):
    """
    Computes the distance matrix for the given number of cells.
    """ 
    total_cells = N * M
    ids = np.arange(total_cells)

    rows = ids // M
    cols = ids % M

    # Reshape for Broadcasting
    r_source = rows.reshape(-1, 1)
    c_source = cols.reshape(-1, 1)
    r_target = rows.reshape(1, -1)
    c_target = cols.reshape(1, -1)

    # 1. Calculate Squared Euclidean Distance (d^2)
    # Rust equivalent: let spatial = (apos.0 - bpos.0).pow(2) + ...
    dist_sq = (r_source - r_target)**2 + (c_source - c_target)**2

    # 2. Apply Obamify's "Strong Penalty" Logic
    # Rust equivalent: (spatial * spatial_weight).pow(2)
    # We square the squared distance again. This creates a distance^4 penalty.
    # We do NOT normalize this (no division by max_dist).
    strong_penalty = dist_sq.astype(np.float32)

    return strong_penalty

def compute_matches(s_bright, s_freq, t_bright, t_freq, dist_mat, alpha=1.0, beta=1.0, gamma=0.1):
    """
    Finds the best UNIQUE match for every target slot.
    - Prevents duplicates (no tile used twice).
    - Maximizes usage of your source library.
    """
    print("Calculating cost matrix for unique matching...")
    start_time = time.time()

    # 1. Reshape for broadcasting (Standard step)
    # Source: Column vector
    sb = s_bright.reshape(-1, 1)
    sf = s_freq.reshape(-1, 1)

    # Target: Row vector
    tb = t_bright.reshape(1, -1)
    tf = t_freq.reshape(1, -1)

    # 2. Calculate the "Cost" (Difference) between every Source and every Target
    # Shape: (Num_Sources, Num_Targets)
    # This matrix can be large! (e.g. 5000x1600)
    diff_b = np.abs(sb - tb)
    diff_f = np.abs(sf - tf)
    cost_matrix = (alpha * diff_b) + (beta * diff_f) + (gamma * dist_mat)

    print("Solving optimal assignment (Hungarian Algorithm)...")
    print("This may take a while. Please wait...")

    # Spinner Logic
    import threading
    import sys
    import itertools
    
    spinner_done = False
    def spin():
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        while not spinner_done:
            sys.stdout.write(next(spinner))   # write the next character
            sys.stdout.flush()                # flush stdout buffer (actual character display)
            sys.stdout.write('\b')            # erase the last written char
            time.sleep(0.1)
    
    t = threading.Thread(target=spin)
    t.start()
    
    try:
        # 3. The Magic Step: Linear Sum Assignment
        # It finds the combination of pairings that minimizes total error
        # while ensuring each Row (Source) is assigned to at most one Col (Target).
        source_indices, target_indices = linear_sum_assignment(cost_matrix)
    finally:
        spinner_done = True
        t.join()

    # 4. Format the result
    # We need an array where result[target_index] = source_index
    # We initialize with -1 in case there are more targets than sources
    matches = np.full(len(t_bright), -1, dtype=int)

    matches[target_indices] = source_indices

    # Fallback: If we have more Targets than Sources, some slots are empty (-1).
    # We fill them with the Greedy match just to prevent black holes.
    if np.any(matches == -1):
        print("Warning: More target slots than source tiles. Filling gaps with duplicates.")
        greedy_matches = np.argmin(cost_matrix, axis=0)
        mask = (matches == -1)
        matches[mask] = greedy_matches[mask]

    print(f"Unique matching complete. Assigned {len(source_indices)} unique tiles. Time: {time.time() - start_time:.3f} seconds ")
    return matches

def reconstruct_mosaic(matches, source_cells, target_shape, N, M):  
    """
    Reconstructs the mosaic from the matches list and source cells.
    """
    # 1. Setup Canvas
    H, W, C = target_shape
    output_img = np.zeros((H, W, C), dtype=np.uint8)

    cell_h = H // N
    cell_w = W // M

    # 2. Iterate through every TARGET SLOT (0 to N*M)
    for tgt_idx, src_idx in enumerate(matches):

        # Calculate where this tile goes (Target Coordinates)
        # Use M (columns) for the modulo/division!
        target_row = tgt_idx // M
        target_col = tgt_idx % M

        y_start = target_row * cell_h
        y_end   = (target_row + 1) * cell_h
        x_start = target_col * cell_w
        x_end   = (target_col + 1) * cell_w

        # 3. Retrieve the Best Matching Source Tile
        # matches[tgt_idx] gives us the index of the source cell we need
        best_source_cell = source_cells[src_idx]

        # 4. Resize and Paste
        # We MUST resize here to ensure the tile fills the slot completely.
        # Even if you resized earlier, integer math might leave 1px gaps.
        slot_h = y_end - y_start
        slot_w = x_end - x_start

        # Resize the source tile to the exact size of the slot
        resized_cell = cv2.resize(best_source_cell, (slot_w, slot_h))

        output_img[y_start:y_end, x_start:x_end] = resized_cell

    return output_img

from matplotlib.collections import EllipseCollection

def render_obamify_voronoi(matches, img_cells, shape, N, M, fps=30, duration=4, render_scale=1.0, hold_duration=1.0, stiffness=0.005, damping=0.98, stretch_factor=0.1, bg_img=None):
    """
    Renders particles with Squash & Stretch deformation based on velocity.
    Uses 'Heavy Overlap' to prevent gaps.
    """
    H, W = shape[:2]
    # In this mode, render_scale affects the PLOT resolution
    
    cell_h = H // N
    cell_w = W // M
    base_size = max(cell_h, cell_w)

    # Calculate frames
    hold_frames = int(hold_duration * fps)
    move_frames = int(duration * fps)
    total_frames = move_frames + hold_frames

    # --- 1. PRE-COMPUTE ---
    print("Pre-computing particle data...")
    particle_colors = []
    positions = []
    velocities = []
    targets = []

    for i, p in enumerate(matches):
        # Target (Destination)
        dst_r, dst_c = divmod(i, M)
        target_y = dst_r * cell_h + cell_h // 2
        target_x = dst_c * cell_w + cell_w // 2

        # Start (Source)
        src_idx = p['src_idx'] if isinstance(p, dict) else p
        src_r, src_c = divmod(src_idx, M)
        start_y = src_r * cell_h + cell_h // 2
        start_x = src_c * cell_w + cell_w // 2

        positions.append([start_x, start_y])
        targets.append([target_x, target_y])
        velocities.append([0.0, 0.0])

        # Color
        tile = img_cells[src_idx]
        avg_color = tile.mean(axis=(0, 1)).astype(np.uint8)
        particle_colors.append(avg_color)

    pos = np.array(positions, dtype=np.float32)
    vel = np.array(velocities, dtype=np.float32)
    tgt = np.array(targets, dtype=np.float32)
    colors = np.array(particle_colors, dtype=np.float32) / 255.0

    # --- 2. SETUP PLOT ---
    print(f"Setting up Squash & Stretch simulation ({len(pos)} particles)...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Background Image
    if bg_img is not None:
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        # Use simple imshow for background. 'extent' ensures it covers the same area as our plot coordinates
        ax.imshow(bg_rgb, extent=[0, W, H, 0])
    else:
        ax.set_facecolor('black')
        
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    
    STIFFNESS = stiffness
    DAMPING = damping
    STRETCH_FACTOR = stretch_factor 

    def update(f):
        nonlocal pos, vel
        # Manual clear for Matplotlib compatibility
        if ax.collections:
            for collection in list(ax.collections):
                collection.remove()
        
        # -- PHYSICS --
        if f >= hold_frames:
            diff = tgt - pos
            acc = (diff * STIFFNESS) - (vel * DAMPING)
            vel += acc
            pos += vel
            
        # -- CALCULATE DEFORMATION --
        speed_sq = np.sum(vel**2, axis=1)
        speed = np.sqrt(speed_sq)
        
        # Angle of velocity
        angles = np.degrees(np.arctan2(vel[:, 1], vel[:, 0]))
        
        # Stretch factor
        stretch = 1.0 + (speed * STRETCH_FACTOR)
        
        # Heavy Overlap to prevent gaps
        OVERLAP = 1.5
        widths  = base_size * stretch * OVERLAP
        heights = (base_size / stretch) * OVERLAP
        
        ec = EllipseCollection(
            widths=widths, 
            heights=heights, 
            angles=angles, 
            units='xy', 
            offsets=pos,
            transOffset=ax.transData,
            facecolors=colors,
            edgecolors='none'
        )
        ax.add_collection(ec)
        return ax.collections

    print("Starting playback with deformation...")
    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000//fps, blit=False)
    plt.show()
    return anim