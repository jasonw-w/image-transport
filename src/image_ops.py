import cv2
import numpy as np

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
