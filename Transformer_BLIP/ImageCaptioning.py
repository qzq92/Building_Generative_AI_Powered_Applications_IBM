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
        blip_model_name (str, optional): Blip model name to use. Defaults to "Salesforce/blip-image-captioning-large" model.

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
    image: Image,
    text_prompt: str,
) -> str:
    """Function which generates captions (guided by additional text prompt if exist) for an given image.

    Args:
        image (Image): Opened Image object
        text_prompt (str): Text string to help guide BLIP model to generate captions.

    Returns:
        str: Returns generated caption by BLIP model. Returns an error string when exception is encountered.
    """
    try:
        blip_model_name = os.environ.get("BLIP_MODEL_NAME")
        processor, model = load_blip_processor_and_model(blip_model_name)
        if not text_prompt:
            inputs = processor(images=image, return_tensors="pt")
        else:
            inputs = processor(images=image, text=str(text_prompt), return_tensors="pt")

        outputs = model.generate(**inputs, max_new_tokens=int(os.environ.get("MODEL_MAX_TOKEN")))

        raw_caption = processor.decode(outputs[0], skip_special_tokens=True)
        return raw_caption
    
    except Exception as err:
        return f"An error occurred in generating caption {err}."

# Test run program module purpose
if __name__ == "__main__":
    load_dotenv()

    img_filepath_eg = os.path.join(os.getcwd(), "images", os.environ.get("VISUALQA_IMAGE_FILENAME"))
    # Open image for processing
    rgb_image = open_image_in_rgb(img_filepath_eg)

    print("Generating prompts with input text_prompt provided...")
    # Input text prompt
    text_prompt = "a photography of"
    if rgb_image:
        caption = generate_caption(
            image=rgb_image,
            text_prompt=text_prompt
        )
        print(caption)
        print()

        # unconditional image captioning
        caption = generate_caption(
            image=rgb_image,
            text_prompt=""
        )
        print("Generating prompts without input text_prompt provided...")
        print(caption)
    
    else:
        print("No image can be found ")