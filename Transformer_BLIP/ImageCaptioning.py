import requests
import os
import PIL
from typing import Tuple
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_blip_processor_and_model(blip_model_name: str = "Salesforce/blip-image-captioning-large") -> Tuple[BlipProcessor,BlipForConditionalGeneration]:
    
    processor = BlipProcessor.from_pretrained(blip_model_name)
    
    model = BlipForConditionalGeneration.from_pretrained(blip_model_name)

    return processor, model


def open_image_in_rgb(img_url: str) -> Image:
    try:
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        return raw_image
    except PIL.UnidentifiedImageError:
        return None

def generate_caption(
        processor: BlipProcessor,
        model: BlipForConditionalGeneration,
        image: Image,
        text_prompt: str = ""
) -> str:

    inputs = processor(image, text_prompt, return_tensors="pt")
    outputs = model.generate(**inputs)

    raw_caption = processor.decode(outputs[0], skip_special_tokens=True)
    return raw_caption

if __name__ == "__main__":
    # Config for raw/blip model
    img_url = os.environ.get("VISUALQA_IMAGE_PATH")
    blip_model_name = os.environ.get("BLIP_MODEL_NAME")

    # Open image for processing
    rgb_image = open_image_in_rgb(img_url=img_url)
    # Conditional captioning
    processor, model = load_blip_processor_and_model(blip_model_name)
    # Input text prompt
    text_prompt = "a photography of"
    caption = generate_caption(
        processor=processor,
        model=model,
        image=rgb_image,
        text_prompt=text_prompt
    )
    print(caption)

    # unconditional image captioning
    caption = generate_caption(
        processor=processor,
        model=model,
        image=rgb_image,
        text_prompt=""
    )
    print(caption)