import cv2
import numpy as np

def normalize_stats(data):
    """
    Normalizes data to a range of [0, 1].
    """
    data = np.array(data)

    if data.ndim == 1:
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val

        if range_val == 0:
            range_val = 1

        return (data - min_val) / range_val

    else:
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        range_vals = maxs - mins

        range_vals[range_vals == 0] = 1

        return (data - mins) / range_vals

def compute_brightness(cells):
    """
    Computes the brightness of each cell.
    """
    averages = []
    for cell in cells:
        if cell.size == 0:
            averages.append(0)
            continue
        
        weighted_sum = np.sum(cell * np.array([0.299, 0.589, 0.114]), axis=(0, 1, 2))
        avg = weighted_sum / (cell.shape[0] * cell.shape[1])
        averages.append(avg)

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
