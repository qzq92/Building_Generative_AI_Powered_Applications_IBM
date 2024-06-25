import requests
import os
import PIL
from typing import Tuple
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv

def load_blip_processor_and_model(blip_model_name: str = "Salesforce/blip-image-captioning-large") -> Tuple[BlipProcessor,BlipForConditionalGeneration]:
    """Function which loads and returns Blip model Processor and Models.

    Args:
        blip_model_name (str, optional): Blip model name to use. Defaults to "Salesforce/blip-image-captioning-large".

    Returns:
        Tuple[BlipProcessor,BlipForConditionalGeneration]: _description_
    """
    processor = BlipProcessor.from_pretrained(blip_model_name)
    
    model = BlipForConditionalGeneration.from_pretrained(blip_model_name)

    return processor, model


def open_image_in_rgb(img_filepath: str) -> Image:
    try:
        raw_image = Image.open(img_filepath).convert('RGB')
        return raw_image
    except PIL.UnidentifiedImageError:
        print("Encountered UnidentifiedImageError in opening the image for processing.")
        return None

def generate_caption(
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    image: Image,
    text_prompt: str = ""
) -> str:
    try:
        inputs = processor(image, text_prompt, return_tensors="pt")
        outputs = model.generate(**inputs)

        raw_caption = processor.decode(outputs[0], skip_special_tokens=True)
        return raw_caption
    
    except Exception as err:
        return f"An error occurred in generating caption {err}."

# Test run program module purpose
if __name__ == "__main__":
    load_dotenv()
    # Config for raw/blip model
    img_filepath_eg = os.path.join(os.getcwd(), "images", os.environ.get("VISUALQA_IMAGE_FILENAME"))
    # Open image for processing
    rgb_image = open_image_in_rgb(img_filepath_eg)

    blip_model_name = os.environ.get("BLIP_MODEL_NAME")
    # Conditional captioning
    processor, model = load_blip_processor_and_model(blip_model_name)
    
    print("Generating prompts with input text_prompt provided...")
    # Input text prompt
    text_prompt = "a photography of"
    if rgb_image:
        caption = generate_caption(
            processor=processor,
            model=model,
            image=rgb_image,
            text_prompt=text_prompt
        )
        print(caption)
        print()

        # unconditional image captioning
        caption = generate_caption(
            processor=processor,
            model=model,
            image=rgb_image,
            text_prompt=""
        )
        print("Generating prompts without input text_prompt provided...")
        print(caption)
    
    else:
        print("No image can be found ")