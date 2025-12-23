import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import EllipseCollection

def render_obamify_voronoi(matches, img_cells, shape, N, M, fps=30, duration=4, render_scale=1.0, hold_duration=1.0, stiffness=0.005, damping=0.98, stretch_factor=0.1, bg_img=None, save_output=None, progress=None):
    """
    Renders particles with Squash & Stretch deformation.
    Optionally saves to 'save_output' path if provided.
    """
    H, W = shape[:2]
    # In this mode, render_scale affects the PLOT resolution
    
    cell_h = H // N
    cell_w = W // M
    base_size = max(cell_h, cell_w)

    # Calculate frames
    total_frames = int(duration * fps)
    hold_frames = int(hold_duration * fps)
    move_frames = max(0, total_frames - hold_frames)

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

    start_pos = pos.copy()
    print(f"Starting simulation... (Total Frames: {total_frames})")

    def update(f):
        nonlocal pos, vel
        
        # RESET STATE at the start of loop (Critical for Save vs Show)
        if f == 0:
            pos[:] = start_pos[:]
            vel[:] = 0.0
        
        if f % 10 == 0:
            print(f"Rendering Frame {f}/{total_frames}...")

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
    
    if save_output:
        print(f"Saving animation to {save_output} (this may take a while)...")
        # Try-catch to handle missing ffmpeg gracefully
        try:
            # Force pillow for GIFs
            writer = 'pillow' if save_output.endswith('.gif') else None
            
            # Custom progress callback
            def progress_callback(current_frame, total_frames_chk):
                if progress:
                    # Map rendering progress from 0.7 to 1.0
                    p = 0.7 + (0.3 * (current_frame / total_frames))
                    progress(p, desc=f"Rendering frame {current_frame}/{total_frames}")

            anim.save(save_output, fps=fps, writer=writer, progress_callback=progress_callback)
            print(f"Success! Saved to {save_output}")
        except Exception as e:
            print(f"Error saving: {e}")
            print("Ensure ffmpeg is installed or try .gif extension.")

    plt.show() # Note: Simulation state is dirty after save, window may look static.
    return anim
