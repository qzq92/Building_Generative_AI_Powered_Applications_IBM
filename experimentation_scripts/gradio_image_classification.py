import gradio as gr
import requests
import os
import torch
from typing import Any, Dict
from PIL import Image
from dotenv import load_dotenv
from torchvision import transforms


def predict(inp: Image) -> Dict:
    # Convert input to tensor and feed to model
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    model = torch.hub.load(
        os.environ.get("TORCH_HUB_MODEL_DIRECTORY"),
        os.environ.get("TORCH_HUB_MODEL_NAME"),
        pretrained=True).eval()
    
    labels = get_imagenet_labels()

    # Disable gradient computation
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences

def get_imagenet_labels() -> list:
    """Function which returns a list of imagenet labels

    Returns:
        list: List of imagenet labels
    """
    response = requests.get("https://git.io/JJkYN")
    labels = response.text.split("\n")

    return labels
       
if __name__ == "__main__":
    load_dotenv()
    # Download human-readable labels for ImageNet.
    response, labels = get_imagenet_labels()
    
    # Define gradio interface class with function to wrap around UI, inputs and outputs gradio. For simplicity, keep the function input limited to image upload. Models used for prediction is controlled by .env.
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
    )
           # Launch interface without creating public link
    iface.launch(
        share = False,
        server_name = os.environ.get("GRADIO_SERVER_NAME"),
        server_port = int(os.environ.get("GRADIO_SERVER_PORT"))
    )