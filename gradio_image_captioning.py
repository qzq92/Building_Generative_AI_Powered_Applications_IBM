import gradio as gr
import os
from Transformer_BLIP.ImageCaptioning import open_image_in_rgb, load_blip_processor_and_model, generate_caption
from dotenv import load_dotenv

if __name__ == "__main__":
    #Load dotenv
    load_dotenv()
    # Construct full filepath
    
    img_filepath_eg = os.path.join(os.getcwd(), "images", os.environ.get("VISUALQA_IMAGE_FILENAME"))
    # Open image for processing
    rgb_image = open_image_in_rgb(img_filepath=img_filepath_eg)

    processor, model = load_blip_processor_and_model(
    blip_model_name = os.environ.get("BLIP_MODEL_NAME")
)

    # Define gradio interface
    iface = gr.Interface(
        fn=generate_caption,
        inputs=[gr.Image(type="pil"), gr.Textbox()],
        outputs="text",
        title="Image Captioning with BLIP",
        description="Upload an image to generate a caption."
    )
    # Launch interface without creating public link
    iface.launch(
        share = False,
        server_name = os.environ.get("GRADIO_SERVER_NAME"),
        server_port = int(os.environ.get("GRADIO_SERVER_PORT"))
    )