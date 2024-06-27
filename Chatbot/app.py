from utils import load_model_tokenizer_and_clean_convo_hist, get_response_for_input
from flask import Flask, request, render_template
from flask_cors import CORS
from flask import request
from dotenv import load_dotenv
import os
import json

# Instantiate Flask with filename as app name
app = Flask(__name__)
# to mitigate CORS errors - a type of error related to making requests to domains other than the one that hosts this webpage.
CORS(app)

# Home page of Flask service to display index.html.
@app.route('/', methods=['GET'])
def home()-> render_template:
    """Function which returns a rendered html template, called only when flask app is launched

    Returns:
        render_template: Rendered html template
    """
    return render_template("index.html")

# THis should not be accessed by web. Serves as a backend service from main page
@app.route('/chatbot', methods=['POST'])
def handle_prompt() -> str:
    """Function which provides a backend service simulating a chatbot service using POST method call from only. No page would be loaded as it is not for access.

    Returns:
        str: Chatbot response to POST request.
    """
    model_name = os.environ.get("CHATBOT_MODEL_NAME")
    # Load model (download on first run and reference local installation for consequent runs)
    hface_auth_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # Load model,tokenizer and convo history
    model, tokenizer, conversation_history = load_model_tokenizer_and_clean_convo_hist(
        model_name = model_name,
        token = hface_auth_token
    )

    # Create conversation history string
    history = "\n".join(conversation_history)

    data = request.get_data(as_text=True)
    data = json.loads(data)
    print(data)
    input_text = data['prompt']

    # Pass into a function containing LLM model/tokenizer, history, and input to retrieve LLM completion.
    response = get_response_for_input(
        tokenizer = tokenizer,
        model = model,
        history = history,
        input_text = input_text)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    load_dotenv()
    app.run(
        debug = True,
        host = os.environ.get("FLASK_RUN_HOST"),
        port = int(os.environ.get("FLASK_RUN_PORT")),
        load_dotenv = True
    )