import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import threading
import sys
import itertools

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
