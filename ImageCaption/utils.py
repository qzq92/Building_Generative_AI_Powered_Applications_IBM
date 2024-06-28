from typing import Tuple
import os 
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def load_blip_processor_and_model(
        blip_model_name: str = "Salesforce/blip-image-captioning-large"
) -> Tuple[BlipProcessor,BlipForConditionalGeneration]:
    """Function which loads and returns Blip model Processor and Models objects based on input blip model name parameter.

    Args:
        blip_model_name (str, optional): Blip model name to use. Defaults to "Salesforce/blip-image-captioning-large" model.

    Returns:
        Tuple[BlipProcessor,BlipForConditionalGeneration]: Pretrained BlipProcessor and BlipForConditionalGeneration classes.
    """
    processor = BlipProcessor.from_pretrained(blip_model_name)
    
    model = BlipForConditionalGeneration.from_pretrained(blip_model_name)

    return processor, model

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
        blip_model_name = os.environ.get("BLIP_MODEL_NAME", default= "Salesforce/blip-image-captioning-large")
        processor, model = load_blip_processor_and_model(blip_model_name)
        if not text_prompt:
            inputs = processor(images=image, return_tensors="pt")
        else:
            inputs = processor(images=image, text=str(text_prompt), return_tensors="pt")

        outputs = model.generate(**inputs, max_new_tokens=int(os.environ.get("MODEL_CAPTION_MAX_TOKEN", default="300")))

        raw_caption = str(processor.decode(outputs[0], skip_special_tokens=True))
        raw_caption = raw_caption.capitalize() + "."
        return raw_caption
    
    except Exception as err:
        return f"An error occurred in generating caption {err}."