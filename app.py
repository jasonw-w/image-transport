import gradio as gr
from src.gradio.img_transport_fn import run_image_transport

def transform_images(source_img, target_img, cell_percentage, quick_gen, brightness_weight, frequency_weight, distance_weight, stiffness, damping, stretch_factor):
    """Simplified version with fewer parameters"""
    
    if source_img is None or target_img is None:
        return None
    
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
            cell_percentage=cell_percentage,
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
        gr.Slider(0.01, 1, value=0.01, step=0.01, label="Cell percentage"),
        gr.Checkbox(value=True, label="Quick Generation"),
        gr.Slider(0.0, 5.0, value=1.0, label="Brightness Weight"),
        gr.Slider(0.0, 5.0, value=1.0, label="Frequency Weight"),
        gr.Slider(0.0, 0.1, value=0.01, label="Distance Weight"),
        gr.Slider(0.0, 0.1, value=0.015, label="Stiffness"),
        gr.Slider(0.8, 1.0, value=0.98, label="Damping"),
        gr.Slider(0.0, 0.5, value=0.1, label="Stretch Factor")
    ],
    outputs=gr.File(label="Download GIF"),
    title="Image Transport Demo",
    description="""
    Create mesmerizing morphing animations between two images using Optimal Transport. Adjust parameters to control the flow, stiffness, and style of the transformation.
    \nDon't like your gif or it doesn't show the whole transformation? No worries, they are saved to cache so you can COOK your perfect image transportation.
    """,
    article="""
    <p style='text-align: center'>Inspired by <a href='https://github.com/jasonw-w/image-transport' target='_blank'>Image Transport</a>. Check out the code and documentation on <a href='https://github.com/jasonw-w/image-transport' target='_blank'>GitHub</a>.</p>
    
    <h3>Parameters Guide</h3>
    <p>Tweaking these values can drastically change the resulting animation.</p>
    
    | Parameter | Default | Description |
    | :--- | :--- | :--- |
    | **Cell Size** | `5` | Resolution of the grid. Smaller = higher detail but slower. |
    | **Quick Gen** | `True` | Faster, lower-FPS preview. Uncheck for full 30 FPS. |
    | **Brightness Weight** | `1.0` | Matches brightness/color intensity. |
    | **Frequency Weight** | `1.0` | Matches texture/edges. |
    | **Distance Weight** | `0.01` | Penalty for moving pixels far. Keeps morph local. |
    | **Stiffness** | `0.005` | Rigidity of the mesh. Higher = solid sheet, Lower = fluid. |
    | **Damping** | `0.98` | How quickly the animation settles (visual bounciness). |
    | **Stretch Factor** | `0.1` | How much the mesh can stretch. |
    """,
    api_name=False
)

if __name__ == "__main__": 
    demo.launch(ssr_mode=False)  # At the bottom