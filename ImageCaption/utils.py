from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Tuple

def load_blip_processor_and_model(blip_model_name: str = "Salesforce/blip-image-captioning-large") -> Tuple[BlipProcessor,BlipForConditionalGeneration]:
    """Function which loads and returns Blip model Processor and Models objects based on input blip model name parameter.

    Args:
        blip_model_name (str, optional): Blip model name to use. Defaults to "Salesforce/blip-image-captioning-large" model.

    Returns:
        Tuple[BlipProcessor,BlipForConditionalGeneration]: Pretrained BlipProcessor and BlipForConditionalGeneration classes.
    """
    processor = BlipProcessor.from_pretrained(blip_model_name)
    
    model = BlipForConditionalGeneration.from_pretrained(blip_model_name)

    return processor, model