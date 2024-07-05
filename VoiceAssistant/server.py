import json
from flask import Flask, render_template, request
from flask_cors import CORS
import os
import numpy as np
import base64
from VoiceAssistant.worker import speech_to_text, text_to_speech, process_message

# Define Flask app
app = Flask(__name__)
# Allow request to be made to different domains
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Home route to return index html interface
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Logic route to process speech input request
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route() -> str:
    """Function which calls the worker's speech_to_text function to perform transcription and returns the transcribed results.

    Returns:
        str: Transcribed speech.
    """
    print("Processing speech-to-text")
    audio_binary = request.data # Get the user's speech from their request
    print("Request...")
    print(request)
    # Call speech_to_text function to transcribe the speech which returns a text string
    text = speech_to_text(audio_binary)

    # Return the response back to the user in JSON format
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    print(response)
    return response

# Logic route to process message
@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage'] # Get user's message from their request
    print('user_message:', user_message)

    # Call openai_process_message function to process the user's message and get a response back
    openai_response_text = process_message(user_message)
    # Clean the response to remove any emptylines
    openai_response_text = os.linesep.join([s for s in openai_response_text.splitlines() if s])

    # Call our text_to_speech function to convert OpenAI Api's reponse to speech
    # The openai_response_speech is a type of audio data, we can't directly send this inside a json as it can only store textual data
    openai_response_speech = text_to_speech(openai_response_text)

    print("Before encoding with base 64 and decode to utf8")
    print(openai_response_speech)
    print(type(openai_response_speech))

    # Lossless commpression with base64 and decode to utf8

    print("Encoding with base64 and decode to utf8")

    if np.any(openai_response_speech):
        print("Decoding text to speech")
        openai_response_speech = base64.b64encode(openai_response_speech).decode('utf-8')
        # Send a JSON response back to the user containing their message's response both in text and speech formats. JSON key is referenced by script.js
        response = app.response_class(
            response=json.dumps({
                "ResponseText": openai_response_text,
                "ResponseSpeech": openai_response_speech
            }),
            status=200,
            mimetype='application/json'
        )
    else:
        # Send a JSON response back to the user containing their message's response both in text only with json.dumps that convert python objects
        response = app.response_class(
            response=json.dumps({
                "ResponseText": "Error encountered in chatbot",
            }),
            status=200,
            mimetype='application/json'
        )

    print(response)
    return response

if __name__ == "__main__":
    app.run(
        debug = True,
        host = os.environ.get("FLASK_RUN_HOST"),
        port = int(os.environ.get("FLASK_RUN_PORT")),
    )