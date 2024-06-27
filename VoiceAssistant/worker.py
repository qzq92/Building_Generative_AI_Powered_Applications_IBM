from openai import OpenAI
from dotenv import load_dotenv
from typing
import requests
import os

load_dotenv()
openai_client = OpenAI(api_key=)


def speech_to_text(audio_binary: str) -> str:
    """The function simply takes audio_binary as the only parameter and then sends it in the body of the HTTP request.

    Args:
        audio_binary (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Set up Watson Speech-to-Text HTTP Api url
    base_url = '...'
    api_url = base_url+'/speech-to-text/api/v1/recognize'

    # Set up parameters for our HTTP reqeust
    params = {
        'model': 'en-US_Multimedia',
    }
    # Set up the body of our HTTP request
    body = audio_binary
    # Send a HTTP Post request
    response = requests.post(url=api_url, params=params, data=audio_binary).json()
    # Parse the response to get our transcribed text
    text = 'null'
    while bool(response.get('results')):
        print('Speech to text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('Recognised text: ', text)
        return text

def text_to_speech(text, voice=""):
    return None


def openai_process_message(user_message):
    # Set the prompt for OpenAI Api
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."
    # Call the OpenAI Api to process our prompt
    openai_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=4000
    )
    print("openai response:", openai_response)
    # Parse the response to get the response message for our prompt
    response_text = openai_response.choices[0].message.content
    return response_text
    return None
