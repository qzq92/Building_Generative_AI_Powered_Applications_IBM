import json
import os
import uuid
import base64
import soundfile as sf

from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask_cors import CORS
from TranslatorAppForEnglish.worker import speech_to_text, text_to_speech, process_message
from language_bark_mapping import language_mapping

# Load env
load_dotenv()

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
def speech_to_text_route():
    audio_binary = request.data # Get the user's speech from their request
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

    language_selected = request.json['languageOption']

    response_text = process_message(
        user_message=user_message,
        language_to_translate_to=language_selected
    )
    # Clean the response to remove any emptylines
    response_text = os.linesep.join([s for s in response_text.splitlines() if s])

    print(f"Responded text: {response_text}")
    # Call our text_to_speech function to convert OpenAI Api's reponse to speech
    # The openai_response_speech is a type of audio data, we can't directly send this inside a json as it can only store textual data
    response_speech, sample_rate = text_to_speech(
        input_text=response_text,
        language_to_translate_to=language_selected
    )

    print("Before encoding with base 64 and decode to utf8")
    print(response_speech)

    # Write ID:
    random_uuid = str(uuid.uuid1())
    random_uuid = random_uuid.replace("-","_")
    audio_file = str(random_uuid) + ".wav"

    # Define save path
    audio_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "bot_audio"
    )
    os.makedirs(audio_dir, exist_ok=True)
    save_audio_path = os.path.join(audio_dir, audio_file)
    print("Writing speech file")
    sf.write(
        file=save_audio_path,
        samplerate=sample_rate,
        data=response_speech
    )

    if response_text != "undefined":
        print("Decoding text to speech")
        with open(save_audio_path, "rb") as binary_file:
            # Read the whole file at once
            data = binary_file.read()
            response_speech = base64.b64encode(data).decode('utf-8')
        # Send a JSON response back to the user containing their message's response both in text and speech formats. JSON key is referenced by script.js
        response = app.response_class(
            response=json.dumps(
                {
                "ResponseText": response_text,
                "ResponseSpeech": response_speech,
                "ResponseID": random_uuid
                }
            ),
            status=200,
            mimetype='application/json'
        )
    else:
        # Send a JSON response back to the user containing their message's response both in text only with json.dumps that convert python objects
        response = app.response_class(
            response=json.dumps({
                "ResponseText": response_text,
                "ResponseID": random_uuid
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