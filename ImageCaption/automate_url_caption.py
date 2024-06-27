import requests
import time
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from utils import load_blip_processor_and_model
from concurrent.futures import ProcessPoolExecutor

def parse_page_for_image_links(url:str) -> list:

    filtered_img_elements = []
    # Download the page
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print('Request failed due to error:', e)


    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all img elements
    img_elements = soup.find_all('img')

    filtered_img_elements = []

    # Iterate over each img element and correct url if possible.
    for img_element in img_elements:
        img_url = img_element.get('src')

        # Skip if the image is an SVG or too small (likely an icon)
        if 'svg' in img_url or '1x1' in img_url:
            continue

        # Correct the URL if it's malformed
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue  # Skip URLs that don't start with http:// or https://
        
        # Add filtered sources to list for tracking
        filtered_img_elements.append(img_url)
    return filtered_img_elements


def generate_caption_for_urls(img_url: list) -> str:
    """Function which generates caption for given url using loaded BLIP processor and model.

    Args:
        img_url (list): _description_

    Returns:
        str: String formatted k:v pair, with k representing image url, and v representing the caption generated.
    """
    # Load the pretrained processor and model
    processor, model = load_blip_processor_and_model(blip_model_name=os.environ.get("BLIP_MODEL_NAME"))
    try:
        # Download the image
        response = requests.get(img_url)
        # Convert the image data to a PIL Image
        raw_image = Image.open(BytesIO(response.content))

        img_res = raw_image.size[0] * raw_image.size[1]
        # Skip very small resolution
        if img_res < int(os.environ.get("MIN_RES_PIXELS")):
            caption = "Image resolution too small to be processed by config.\n"
        else:
            raw_image = raw_image.convert('RGB')

            # Process the image
            inputs = processor(raw_image, return_tensors="pt")
            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(out[0], skip_special_tokens=True)

        output_str = f"{img_url}: {caption}.\n"
        return output_str
    
    # Catch all general exception with common output string construct
    except Exception as e:
        error_caption = f"Error processing image file due to {e}."
        output_str = f"{img_url}: {error_caption}"
        return output_str

if __name__ == "__main__":
    # Load sys environment
    load_dotenv()
    # URL of the page to scrape
    url = os.environ.get("IMAGES_SOURCE_URL")
    filtered_img_url_list = parse_page_for_image_links(url = url)
    print(f"Number of images to be captioned: {len(filtered_img_url_list)}")

    # Set optimal max threads (reserved 2 threads for cpu to cater to other uses)
    MAX_THREADS = min(os.cpu_count() - 2, len(filtered_img_url_list))
    
    # Generate captions for each image url and save it into a text file in a defined filepath.
    parent_dir = os.environ.get("PYTHONPATH")
    if not parent_dir:
        parent_dir = os.getcwd()

    caption_filepath = os.path.join(parent_dir,"generated_captions.txt")
    with open(caption_filepath, "w") as caption_file:
        print(f"Using multiprocessing with {MAX_THREADS} threads to generate captions")
        time_start = time.time()
        with ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
            captions_list = list(executor.map(generate_caption_for_urls, filtered_img_url_list))
            print(captions_list)
        time_end = time.time()
        print(f"Time elapsed: {round(time_end - time_start, 2)}s")
        print("Writing captions to file ...")
        for caption in captions_list:
            caption_file.write(caption.capitalize()+"\n")