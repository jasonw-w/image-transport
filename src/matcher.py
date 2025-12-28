import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import threading
import sys
import os
import itertools

def compute_distance_matrix(N, M):
    """
    Computes the distance matrix for the given number of cells.
    """ 
    total_cells = N * M
    ids = np.arange(total_cells)

    rows = ids // M
    cols = ids % M

    r_source = rows.reshape(-1, 1)
    c_source = cols.reshape(-1, 1)
    r_target = rows.reshape(1, -1)
    c_target = cols.reshape(1, -1)

    dist_sq = (r_source - r_target)**2 + (c_source - c_target)**2

    strong_penalty = dist_sq.astype(np.float32)

    return strong_penalty

def compute_matches(s_bright, s_freq, t_bright, t_freq, dist_mat, alpha=1.0, beta=1.0, gamma=0.1):
    """
    Finds the best UNIQUE match for every target slot.
    - Prevents duplicates (no tile used twice).
    - Maximizes usage of your source library.
    """
    cache_dir = "cache_matches"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    input_data = [s_bright, s_freq, t_bright, t_freq, dist_mat, alpha, beta, gamma]
    input_bytes = b"".join([x.tobytes() if isinstance(x, np.ndarray) else str(x).encode() for x in input_data])
    import hashlib
    file_hash = hashlib.md5(input_bytes).hexdigest()
    cache_file = os.path.join(cache_dir, f"match_{file_hash}.npy")

    if os.path.exists(cache_file):
        print(f"Loading cached matches from {cache_file}")
        return np.load(cache_file)

    print("Calculating cost matrix for unique matching")
    start_time = time.time()

    sb = s_bright.reshape(-1, 1)
    sf = s_freq.reshape(-1, 1)

    tb = t_bright.reshape(1, -1)
    tf = t_freq.reshape(1, -1)

    diff_b = np.abs(sb - tb)
    diff_f = np.abs(sf - tf)
    cost_matrix = (alpha * diff_b) + (beta * diff_f) + (gamma * dist_mat)

    print("Solving optimal assignment (Hungarian Algorithm)")
    print("This may take a while. Please wait")

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
        source_indices, target_indices = linear_sum_assignment(cost_matrix)
    finally:
        spinner_done = True
        t.join()

    matches = np.full(len(t_bright), -1, dtype=int)

    matches[target_indices] = source_indices

    if np.any(matches == -1):
        print("Warning: More target slots than source tiles. Filling gaps with duplicates.")
        greedy_matches = np.argmin(cost_matrix, axis=0)
        mask = (matches == -1)
        matches[mask] = greedy_matches[mask]

    print(f"Saving matches to {cache_file}")
    np.save(cache_file, matches)

    files = sorted(
        [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.npy')],
        key=os.path.getmtime
    )
    
    while len(files) > 5:
        oldest_file = files.pop(0)
        print(f"Cache limit exceeded. Deleting oldest file: {oldest_file}")
        try:
            os.remove(oldest_file)
        except OSError as e:
            print(f"Error deleting {oldest_file}: {e}")

    print(f"Unique matching complete. Assigned {len(source_indices)} unique tiles. Time: {time.time() - start_time:.3f} seconds ")
    return matches
