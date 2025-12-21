import cv2
import numpy as np
import os

# New Modules
from src import image_ops
from src import features
from src import matcher
from src import visualizer

# --- CONFIGURATION ---
SOURCE_PATH = "demo/imgs/nature.jpg"
TARGET_PATH = "demo/imgs/obama.png"
CELL_SIZE = 1 # 10 is safer for memory, 2 is high quality but risky

# Export Settings
EXPORT_MP4 = True  # Set to True to save animation
OUTPUT_FILENAME = "simulation.gif"

# --- 1. LOAD & PREPARE ---
src_img = cv2.imread(SOURCE_PATH) 
tgt_img = cv2.imread(TARGET_PATH)

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
            alpha=5.0,
            beta=1.0,
            gamma=0.01)
        np.save(CACHE_FILE, matches)
else:
    matches = matcher.compute_matches(
        src_brightness, src_frequency, 
        tgt_brightness, tgt_frequency, 
        dist_mat,
        alpha=5.0,
        beta=1.0,
        gamma=0.01)
    np.save(CACHE_FILE, matches)
    print(f"Saved matches to {CACHE_FILE}")

# --- 5. RENDER ---
# Optional: Static reconstruction
# mysaic = image_ops.reconstruct_mosaic(matches, src_cells, src_img.shape, N, M) 

# Animation
output_file = visualizer.render_obamify_voronoi(
    matches, src_cells, src_img.shape, N, M, 
    fps=30, 
    duration=20, 
    render_scale=1.0, 
    hold_duration=0.25, 
    stiffness=0.01, 
    damping=0.98,
    stretch_factor=0.1,
    bg_img=src_img,
    save_output=OUTPUT_FILENAME if EXPORT_MP4 else None)
