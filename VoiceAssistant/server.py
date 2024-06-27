import base64
import json
from flask import Flask, render_template, request
from worker import speech_to_text, text_to_speech, openai_process_message
from flask_cors import CORS
import os

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
    return None

# Logic route to process message
@app.route('/process-message', methods=['POST'])
def process_prompt_route():
    response = app.response_class(
        response=json.dumps(
            {"openaiResponseText": None, "openaiResponseSpeech": None}
        ),
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
