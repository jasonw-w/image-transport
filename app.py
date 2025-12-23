import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gradio as gr
from src.gradio.img_transport_fn import run_image_transport

DESCRIPTION = """
Transform a source image to match a target image with an animated morphing effect! 

**How it works:**
1. Upload a source image (or use an example)
2. Upload a target image (or use an example)  
3. Adjust parameters or use defaults
4. Click "Generate Transform" and wait (this may take 1-3 minutes)
5. Download your animated GIF!

**Tips:**
- Start with "Quick Generation" enabled for faster results
- Smaller cell sizes = faster processing but lower quality
- Larger cell sizes = slower but more detailed animation
"""

ARTICLE = """
### About
Inspired by the [obamify project](https://github.com/zach-snell/obamify), this tool uses optical transport to morph images. 

**Built with:**  OpenCV, NumPy, Matplotlib, Gradio

**GitHub:** [jasonw-w/image-transport](https://github.com/jasonw-w/image-transport)
"""

def transform_images(
    source_img,
    target_img,
    brightness_weight,
    frequency_weight,
    distance_weight,
    stiffness,
    damping,
    stretch_factor,
    cell_size,
    quick_gen,
    progress=gr.Progress()
):
    """Gradio interface function"""
    
    if source_img is None or target_img is None:
        raise gr.Error("Please upload both source and target images!")
    
    try:
        output_path = run_image_transport(
            source_image=source_img,
            target_image=target_img,
            brightness_weight=brightness_weight,
            frequency_weight=frequency_weight,
            distance_weight=distance_weight,
            stiffness=stiffness,
            damping=damping,
            stretch_factor=stretch_factor,
            cell_size=cell_size,
            quick_gen=quick_gen,
            progress=progress
        )
        
        return output_path
        
    except Exception as e:
        raise gr.Error(f"Error during transformation: {str(e)}")

demo = gr.Interface(
    fn=transform_images,
    inputs=[
        gr.Image(type="pil", label="Source Image"),
        gr.Image(type="pil", label="Target Image"),
        gr.Slider(0, 2, value=1.0, label="Brightness Weight"),
        gr.Slider(0, 2, value=1.0, label="Frequency Weight"),
        gr.Slider(0, 1, value=0.001, label="Distance Weight"),
        gr.Slider(0, 0.1, value=0.01, label="Stiffness(increase if only the first bit of the gif is played)"),
        gr.Slider(0, 2, value=0.98, label="Damping"),
        gr.Slider(0, 1, value=0.1, label="Stretch Factor"),
        gr.Slider(1, 20, value=1, step=1, label="Cell Size"),
        gr.Checkbox(value=True, label="Quick Generation")
    ],
    outputs=gr.Image(type="filepath", label="Output Animation"),
    title="Image Morphing Transport",
    description=DESCRIPTION,
    article=ARTICLE
)

if __name__ == "__main__":
    demo.launch()