import gradio as gr
import os
from Transformer_BLIP.ImageCaptioning import open_image_in_rgb, load_blip_processor_and_model, generate_caption
from dotenv import load_dotenv

if __name__ == "__main__":
    #Load dotenv
    load_dotenv()
    
    processor, model = load_blip_processor_and_model(
    blip_model_name = os.environ.get("BLIP_MODEL_NAME")
)

    # Define gradio interface
    iface = gr.Interface(
        fn=generate_caption,
        inputs=[gr.Image(type="pil", show_label=False), gr.Textbox(placeholder="Text prompt to guide caption generation", label="Text Prompt")],
        outputs=gr.Textbox(label="Generated Caption"),
        title="Image Captioning with BLIP Model",
        description="Upload an image file and input text (optional) in the text bar under 'text_prompt' to generate a caption."
    )
    # Launch interface without creating public link
    iface.launch(
        share = False,
        server_name = os.environ.get("GRADIO_SERVER_NAME"),
        server_port = int(os.environ.get("GRADIO_SERVER_PORT")),
    )