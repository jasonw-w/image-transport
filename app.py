import gradio as gr
from src.gradio.img_transport_fn import run_image_transport

def transform_images(source_img, target_img, cell_size, quick_gen):
    """Simplified version with fewer parameters"""
    
    if source_img is None or target_img is None:
        return None
    
    try: 
        output_path = run_image_transport(
            source_image=source_img,
            target_image=target_img,
            brightness_weight=1.0,
            frequency_weight=1.0,
            distance_weight=0.01,
            stiffness=0.005,
            damping=0.98,
            stretch_factor=0.1,
            cell_size=int(cell_size),
            quick_gen=quick_gen,
            progress=None
        )
        
        return output_path
        
    except Exception as e:
        return f"Error:  {str(e)}"

# Simple interface
demo = gr.Interface(
    fn=transform_images,
    inputs=[
        gr.Image(label="Source Image", type="pil"),
        gr.Image(label="Target Image", type="pil"),
        gr.Slider(1, 20, value=5, step=1, label="Cell Size"),
        gr.Checkbox(value=True, label="Quick Generation")
    ],
    outputs=gr.File(label="Download GIF"),
    title="🎨 Image Transport Demo",
    description="Transform images with animated morphing effects! ",
    article="Based on obamify.  Upload two images and click submit."
)

if __name__ == "__main__": 
    demo.launch()