import functions as f
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment

src_img = cv2.imread("imgs/source.png") 
tgt_img = cv2.imread("imgs/target (1).png")

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
