# Simple Generative AI Powered Applications

Repository containing codebase covering various GenAI module applications based on "Building Generative AI-Powered Applications with Python" Coursera Course organised by IBM. 

1. Image Captioning
    - Gradio Interface UI for uploading image to perform captioning
    - Python script for generating captions on all available images retrieved from a specified UI.
2. Simple Chatbot

## Environment file to edit

Please create an *.env* file with the following parameters.

```
HUGGINGFACEHUB_API_TOKEN = <Your HuggingFaceHub API Token>
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"

# Required if automate_url_caption.py script is run.
IMAGES_SOURCE_URL = <URL containing images>
PYTHONPATH = <Path to this repository which is downloaded>
MODEL_MAX_TOKEN = <Max token allowed for LLM completion for all LLM models> 

# Condition defining captions to be generated for images above specific resolution
MIN_RES_PIXELS = <Number of images pixels required to allow captions to be generated>

# Required if you are running *imagecaptioning.py* under *ImageCaption/* folder
VISUALQA_IMAGE_FILENAME = <Files under *images* subfolder> #e.g. "demo_image.jpg" 

# For Image classification **(Required if you are running the gradio_image_classification.py for image classification models)**
TORCH_HUB_MODEL_DIRECTORY = "pytorch/vision:v0.6.0"
TORCH_HUB_MODEL_NAME = <Torch Hub Model name> #eg resnet18

# For Chatbot. Please select a model that can be executed on your computer.
CHATBOT_MODEL_NAME = "facebook/blenderbot-400M-distill"

# Gradio Config for Server and Port.
GRADIO_SERVER_NAME = <Name of DNS Resolvable Server or IP Address> # Eg "127.0.0.1"
GRADIO_SERVER_PORT = <Your preferred port> # Eg "7860"

# FLASK CONFIG. SERVER_NAME DEFAULTS TO 127.0.0.1 if empty. SERVER_PORT DEFAULTS to 5000 if empty.
FLASK_SERVER_NAME = "127.0.0.1"
FLASK_SERVER_PORT = "5001"
```

## Installation and execution

Please use Anaconda distribution to install the necessary libraries with the following command

```
conda env create -f environment.yml
```

Upon installation and environment exectuion, run the following command to start Gradio interface.

```
python ImageCaption/run_gradio_image_upload_captioning
```

You should see a Gradio UI as follows:

![SampleUI](images/SampleUI.png)

**A working example with generated caption**

![SampleWorkingExample](images/SampleUI_w_Caption.png)

### For experimentation purpose with caption models generated output without Gradio

Please run the following command in the repository main folder

```
python Transformer_BLIP/ImageCaptioning.py
```

## Programming languages/tools involved
- Python
- Flask
- Gradio
    - Interface
    - Textbox
    - Image
- HuggingFace
    - Transformer models involving BlipProcessor, BlipForConditionalGeneration
- Concurrence library
    - Multiprocessing with 10 threads for image captioning
        - 14 images took 254 seconds
## Acknowledgement and Credits

The codebase for the simple apps developed are referenced from *"Building Generative AI-Powered Applications with Python"* by IBM available at https://www.coursera.org/learn/building-gen-ai-powered-applications, and also IBM's LLM Application Chatbot Github Repository for the webpage template provided for the Chatbot module, accessible https://github.com/ibm-developer-skills-network/LLM_application_chatbot.