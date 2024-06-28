import os
import PIL
from PIL import Image
from utils import generate_caption
from dotenv import load_dotenv

def open_image_in_rgb(img_filepath: str) -> Image:
    try:
        raw_image = Image.open(img_filepath).convert('RGB')
        return raw_image
    except PIL.UnidentifiedImageError:
        print("Encountered UnidentifiedImageError in opening the image for processing.")
        return None

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
        caption = "No image can be found, hence no caption would be generated."
        print(caption)