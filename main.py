import functions as f
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment

# Ensure imgs directory exists (using absolute path for consistency)
base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, 'imgs')
os.makedirs(img_dir, exist_ok=True)

src_path = os.path.join(img_dir, 'source.png')
tgt_path = os.path.join(img_dir, 'target (1).png')

src_img = cv2.imread(src_path) 
tgt_img = cv2.imread(tgt_path)

# Check if images were loaded successfully
if src_img is None:
    raise OSError(f"Error: Could not load '{src_path}'. Please add a source image to the imgs directory.")
if tgt_img is None:
    raise OSError(f"Error: Could not load '{tgt_path}'. Please add a target image to the imgs directory.")

tgt_img = f.match_aspect_ratio(src_img, tgt_img)

cell_size = 1
N = src_img.shape[0]//cell_size
M = src_img.shape[1]//cell_size
src_cells = f.split(N, M, src_img)
tgt_cells = f.split(N, M, tgt_img)

src_brightness = f.compute_brightness(src_cells)
tgt_brightness = f.compute_brightness(tgt_cells)

src_frequency = f.compute_frequency(src_cells)
tgt_frequency = f.compute_frequency(tgt_cells)

dist_mat = f.compute_distance_matrix(N, M) 


# Check for cached matches to save time
CACHE_FILE = "matches_cache.npy"

if os.path.exists(CACHE_FILE):
    print(f"Loading matches from {CACHE_FILE}...")
    matches = np.load(CACHE_FILE)
else:
    matches = f.compute_matches(
        src_brightness, src_frequency, 
        tgt_brightness, tgt_frequency, 
        dist_mat,
        alpha=5.0,
        beta=1.0,
        gamma=0.01)
    np.save(CACHE_FILE, matches)
    print(f"Saved matches to {CACHE_FILE}")

img = f.reconstruct_mosaic(matches, src_cells, src_img.shape, N, M) 

output_file = f.render_obamify_voronoi(
    matches, src_cells, src_img.shape, N, M, 
    fps=30, 
    duration=15, 
    render_scale=1.0, 
    hold_duration=0.5, 
    stiffness=0.005, 
    damping=0.98,
    stretch_factor=0.1,
    bg_img=src_img)
