# Simple Generative AI Powered Applications

Repository containing codebase covering various GenAI module applications based on "Building Generative AI-Powered Applications with Python" Coursera Course organised by IBM. 

1. Image Captioning
    - Gradio Interface UI for uploading image to perform captioning
    - Python script for generating captions on all available images retrieved from a specified UI.

2. Simple Chatbot
    - Frontend interface supported by HTML, Javascript and Flask
    - Backend chat service supported by the use of HuggingFaceHub model loaded into PC.

## Environment file to create and edit

Include the following parameters. Please enter your API Token keys where necessary.

```
OPENAI_API_KEY = <YOUR API TOKEN>
OPENAI_MAX_TOKEN = "4000"
OPENAI_MODEL_NAME = "gpt-3.5-turbo"
HUGGINGFACEHUB_API_TOKEN = <YOUR API TOKEN>
PYTHONPATH =

# For Image Captioning
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"
VISUALQA_IMAGE_FILENAME = "demo_image.jpg" ## For imagecaptioning.py use
IMAGES_SOURCE_URL = "https://en.wikipedia.org/wiki/IBM" ## For automate_url_caption.py use
MIN_RES_PIXELS = "400"
MODEL_CAPTION_MAX_TOKEN = "300"

# For Chatbot. Please select a model that can fit and run on your computer.
CHATBOT_MODEL_NAME = "facebook/blenderbot-400M-distill"
TEMPERATURE = "0.5" # Anything above 0 but less than 1
MODEL_CHATBOT_MAX_LENGTH = "80"

# For VoiceAssistant:
# Refer to https://huggingface.co/models?pipeline_tag=automatic-speech-recognition. For long form transcription, please use "distil-whisper/distil-large-v3"
HUGGINGFACE_STT_MODEL_NAME = "openai/whisper-small"

# Refer to models page https://huggingface.co/models?pipeline_tag=text-to-speech
HUGGINGFACE_TTS_MODEL_NAME  = "microsoft/speecht5_tts"

# Gradio Config for Server and Port.
GRADIO_SERVER_NAME = <Name of DNS Resolvable Server or IP Address> #E.g "127.0.0.1"
GRADIO_SERVER_PORT = <Your preferred port> #E.g "7860"

# FLASK CONFIG. SERVER_NAME DEFAULTS TO 127.0.0.1 if empty. SERVER_PORT DEFAULTS to 5000 if empty.
FLASK_RUN_HIST = <Host Name/IP> #E.g "127.0.0.1"
FLASK_RUN_PORT = <Your preferred port> #E.g "7860"
```

Corresponding Javascipt to be edited (For chatbot app only)

```
async function makePostRequest(msg) {
    const url = "http://<Flask Server Name>:<Port>/chatbot";  // Make a POST request to this url
    const requestBody = {
      prompt: msg
    };
```

## Installation and execution

Please use Anaconda distribution to install the necessary libraries with the following command

```
conda env create -f environment.yml
```

Upon installation and environment exectuion, please run the relevant command based on the app required to run.

### 1. Image Captioning

```
cd ImageCaption/
python run_gradio_image_upload_captioning.py
```

You should see a Gradio UI as follows:

![SampleImageCaptionUI](images/SampleImageCaptionUI.png)

**A working example with generated caption**

![SampleImageCaptionWorkingExample](images/SampleImageCaptionUI_working.png)

** For experimentation purpose with caption models generated output without Gradio **

Please run the following command in the repository main folder

```
cd ImageCaption/
python imagecaptioning.py
```

### 2. Simple Chatbot

Suggested chatbot model from HuggingFace that can be loaded on to your PC would be *facebook/blenderbot-400M-distill*. It is known to outperforms existing models in terms of longer conversations over multiple sessions and is more knowledgeable and has more factual consistency, according to human evaluators. (Source: [ParlAI](https://parl.ai/projects/blenderbot2/#:~:text=A%20chatbot%20with%20its%20own,consistency%2C%20according%20to%20human%20evaluators.))

**Disclaimer: You may need to configure *TEMPERATURE* environment to control chatbot responses. As this is just a simple project, the chatbot is not meant to be provide perfectly great responses and the result of such is largely dependent on the input chat message provided and other model configurations.**

```
cd Chatbot/
python app.py
```

OR
```
cd Chatbot/
flask run -h <host Name/IP> -p <port>
```

You should see a sample chatbot interface below:
![SampleChatbotUI](images/SampleChatbotUI.png)

A demonstration example of how conversation would be as follows:
![SampleChatbotConversation](images/SampleChatBotInteraction.png)

To terminate program, press 'Ctrl' + 'C'.

**Testing of chatbot response with curl**
Ensure that you have executed above command to get flask running. Then execute an example command below

```
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you today?"}' <Flask Server Host>:<Port>/chatbot
```
## 3. Voice Assistant


### 3A. Simple transcription service with OpenAI model experimentation setup via Gradio Frontend

You may can either upload your own mp3 file or use a sample mp3 file provided which are sourced from

1) **Archived LiveATC Recordings** [link](https://www.liveatc.net/recordings.php) - Expect poor performance due to background noise 

2) **Ted Talks Daily Trailer** [link](https://audiocollective.ted.com/#shows-1) - Expect good performance due to clear audio, without background noise.

Run the following command in the repository

```
cd VoiceAssistant/experimentations
python gradio_interface.py
```

Access the Gradio Interface via the host IP/Port specified as seen below:
![SampleTranscriptionService](images/SampleGradioTranscriptionUI.png)

Sample audio transcription from file:
![SampleAudioTranscription](images/SampleAudioFileTranscription.png)

Disclaimer: Do expect transcription in accuracies as results are largely dependent on the quality and length of audio file.  

### 3.1 Place RootCA cert in the certs folder

This is a prerequisite for docker build process listed in 3.2.

If you are in Linux machine, do the following
```
cp /usr/local/share/ca-certificates/rootCA.crt /home/project/chatapp-with-voice-and-openai/certs/
```

For Windows, refer to the steps for extracting certificates.

RootCA extraction steps for Windows reference:
(https://help.zscaler.com/deception/exporting-root-ca-certificate-active-directory-certificate-service)

In particular, the specific cert required is illustrated below

The cert to export is highlighted below:
![RootCert](images/RootCert_Export_Windows.png)

### 3.2 Run docker image with the following (build/rebuild if needed)

Ensure that your docker engine is active.

```
docker build . -t voice-chatapp-powered-by-openai
docker run -p 8000:8000 voice-chatapp-powered-by-openai
```



## Programming languages/tools involved
- Python
- JavaScript
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

The codebase for the simple apps developed are referenced from *"Building Generative AI-Powered Applications with Python"* by IBM available at https://www.coursera.org/learn/building-gen-ai-powered-applications.

Additional acknowledgement for different sections:

Chatbot module webpage template: [IBM's LLM Application Chatbot Github Repository](https://github.com/ibm-developer-skills-network/LLM_application_chatbot)

Voice assistant webpage template: [Arora-R](https://github.com/arora-r/chatapp-with-voice-and-openai-outline)

RootCert Export: [RootCert-Export steps for Windows](https://help.zscaler.com/deception/exporting-root-ca-certificate-active-directory-certificate-service)

OpenAI Speech to text: [Speech-to-text](File uploads are currently limited to 25 MB and the following input file types are supported: mp3, mp4, mpeg, mpga, m4a, wav, and webm)