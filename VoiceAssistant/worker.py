from openai import OpenAI
from dotenv import load_dotenv
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing
import requests
import os

load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def speech_to_text(audio_binary: list) -> str:
    """The function simply takes audio_binary as the only parameter and then sends it in the body of the HTTP request.

    Args:
        audio_binary (str): Array of values sampled from audio.

    Returns:
        str: _description_
    """
    print(audio_binary)
    model = WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=os.environ.get("STT_MODEL_NAME")
    )
    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=os.environ.get("STT_MODEL_NAME")
    )

    input_features = processor(
        audio_binary, sampling_rate=44100, return_tensors="pt"
    ).input_features 


    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )
    print(transcription)

    return transcription[0]

def text_to_speech(text: str):

    # Set the headers for our HTTP request
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }
    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }
    # Send a HTTP Post request to Watson Text-to-Speech Service
    # response = requests.post(url=api_url, headers=headers, json=json_data)
    print('Text to speech response:', response)

    # Return audio data
    return response.content

def openai_process_message(user_message: str) -> str:
    """Function which processes user message with OpenAI models and generates completion as response.

    Args:
        user_message (str): User provided message in string.

    Returns:
        str: OpenAI LLM model completion string.
    """
    # Set the prompt for OpenAI Api
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."
    # Call the OpenAI Api to process our prompt
    openai_response = openai_client.chat.completions.create(
        model = os.environ.get("OPENAI_MODEL_NAME"),
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens = os.environ.get("OPENAI_MAX_TOKEN")
    )
    print("openai response:", openai_response)
    # Parse the response to get the response message for our prompt
    response_text = openai_response.choices[0].message.content
    return response_text