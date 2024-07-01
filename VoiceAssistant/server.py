import json
import base64
from flask import Flask, render_template, request
from worker import speech_to_text, text_to_speech, openai_process_message
from flask_cors import CORS
import os

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
    print(audio_binary)
    print(type(audio_binary))
    # Call speech_to_text function to transcribe the speech
    text = speech_to_text(audio_binary)
    # Return the response back to the user in JSON format
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    print(response)
    response = "Testing in progress"
    return response

# Logic route to process message
@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage'] # Get user's message from their request
    print('user_message', user_message)

    # Call openai_process_message function to process the user's message and get a response back
    openai_response_text = openai_process_message(user_message)
    # Clean the response to remove any emptylines
    openai_response_text = os.linesep.join([s for s in openai_response_text.splitlines() if s])

    print(f"Generated ChatModel response: {openai_response_text}. Synthesizing to speech...")
    # Call our text_to_speech function to convert OpenAI Api's reponse to speech
    # The openai_response_speech is a type of audio data, we can't directly send this inside a json as it can only store textual data
    openai_response_speech = text_to_speech(openai_response_text)

    print(openai_response_speech)
    # # convert openai_response_speech to base64 string so it can be sent back in the JSON response
    print("Encdoing with base64 and decode to utf8")
    openai_response_speech = base64.b64encode(openai_response_speech).decode('utf-8')
    # Send a JSON response back to the user containing their message's response both in text and speech formats
    if "error" in dict(openai_response_speech):
        print("Error in generating synthesizing text to speech. Response will not have any speech")
    
        response = app.response_class(
            response=json.dumps({
                "openaiResponseText": openai_response_text,
            }),
            status=200,
            mimetype='application/json'
        )
    else:
        response = app.response_class(
            response=json.dumps({
                "openaiResponseText": openai_response_text,
                "openaiResponseSpeech": openai_response_speech
            }),
            status=200,
            mimetype='application/json'
        )
    return response

if __name__ == "__main__":
    app.run(
        debug = True,
        host = os.environ.get("FLASK_RUN_HOST"),
        port = int(os.environ.get("FLASK_RUN_PORT")),
        load_dotenv = True
    )
