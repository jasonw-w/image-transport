import cv2
import gradio as gr
import numpy as np
import os
import tempfile
from pathlib import Path

def run_image_transport(
    source_image,
    target_image,
    brightness_weight=1.0,
    frequency_weight=1.0,
    distance_weight=0.01,
    stiffness=0.005,
    damping=0.98,
    stretch_factor=0.1,
    cell_percentage=0.01,
    quick_gen=True,
    progress=None,
    src_fit_target=False
):
    """
    Wrapper function for Gradio demo.
    
    Args:
        source_image:  PIL Image or numpy array
        target_image: PIL Image or numpy array
        ...  (other parameters)
        progress:  Gradio progress callback (optional)
        
    Returns:
        str: Path to generated GIF file
    """
    from src import image_ops, features, matcher, visualizer
    
    # Update progress
    if progress:
        progress(0.1, desc="Loading images...")
    
    # Convert PIL to numpy if needed
    if hasattr(source_image, 'convert'):
        source_image = np.array(source_image.convert('RGB'))
        source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    if hasattr(target_image, 'convert'):
        target_image = np.array(target_image.convert('RGB'))
        target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
    
    # Save to temp files
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as src_tmp:
        cv2.imwrite(src_tmp.name, source_image)
        source_path = src_tmp.name
        
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tgt_tmp:
        cv2.imwrite(tgt_tmp.name, target_image)
        target_path = tgt_tmp.name
    
    try:
        # Load images
        src_img = cv2.imread(source_path)
        tgt_img = cv2.imread(target_path)
        
        if progress:
            progress(0.2, desc="Resizing images...")
        
        # Resize source to match target
        if src_fit_target:  
            src_img = image_ops.match_aspect_ratio(tgt_img, src_img)
            CELL_SIZE = (src_img.shape[0] * cell_percentage + src_img.shape[1] * cell_percentage) // 2
        else:
            tgt_img = image_ops.match_aspect_ratio(src_img, tgt_img)
            CELL_SIZE = (tgt_img.shape[0] * cell_percentage + tgt_img.shape[1] * cell_percentage) // 2
        
        print(f"Cell Size: {CELL_SIZE}")    

        # Grid splitting
        N = int(src_img.shape[0] // CELL_SIZE)
        M = int(src_img.shape[1] // CELL_SIZE)
        
        if N < 1 or M < 1:
            raise ValueError("Cell size is too large. Try a smaller value.")
        
        print(f"Grid Size: {N}x{M} = {N*M} cells")
        
        if N*M > 50000:
            raise ValueError("Too many cells!  This will take too long.  Try increasing cell_size.")
        
        src_cells = image_ops.split(N, M, src_img)
        tgt_cells = image_ops.split(N, M, tgt_img)
        
        if progress:
            progress(0.3, desc="Extracting features...")
        
        # Feature extraction
        src_brightness = features.compute_brightness(src_cells)
        tgt_brightness = features.compute_brightness(tgt_cells)
        src_frequency = features.compute_frequency(src_cells)
        tgt_frequency = features.compute_frequency(tgt_cells)
        
        if progress:
            progress(0.5, desc="Computing matches (this may take a while)...")
        
        # Matching
        dist_mat = matcher.compute_distance_matrix(N, M)
        matches = matcher.compute_matches(
            src_brightness, src_frequency,
            tgt_brightness, tgt_frequency,
            dist_mat,
            alpha=brightness_weight,
            beta=frequency_weight,
            gamma=distance_weight
        )
        
        if progress:
            progress(0.7, desc="Rendering animation...")
        
        # Render
        output_file = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
        
        if quick_gen:
            fps = 15
            render_scale = 0.5
            hold_duration = 0.5
            duration = 20
        else:
            fps = 30
            duration = 20
            render_scale = 1.0
            hold_duration = 0.25
        
        visualizer.render_obamify_voronoi(
            matches, src_cells, src_img.shape, N, M,
            fps=fps,
            duration=duration,
            render_scale=render_scale,
            hold_duration=hold_duration,
            stiffness=stiffness,
            damping=damping,
            stretch_factor=stretch_factor,
            bg_img=src_img,
            save_output=output_file,
            progress=progress
        )
        
        if progress:
            progress(1.0, desc="Complete!")
        
        return output_file
        
    finally:
        try:
            os.unlink(source_path)
            os.unlink(target_path)
        except:
            pass