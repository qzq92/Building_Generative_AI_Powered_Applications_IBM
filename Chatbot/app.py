from flask import Flask, request, render_template
from flask_cors import CORS
from flask import request
from dotenv import load_dotenv
from utils import load_model_tokenizer_and_clean_convo_hist
import os
import json

# Instantiate Flask with filename as app name
app = Flask(__name__)
# to mitigate CORS errors - a type of error related to making requests to domains other than the one that hosts this webpage.
CORS(app)

# Landing page of Flask service.
@app.route('/chatbot', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/chatbot', methods=['POST'])
def handle_prompt() -> str:
    """Function which serves as a Flask backend service simulating a chatbot service accessible by '/chatbot' route.

    Returns:
        str: Chatbot response to POST request.
    """
    # Read prompt as text from HTTP request body, eg curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you today?"}' 127.0.0.1:5000/chatbot
    data = request.get_data(as_text=True)
    data = json.loads(data)

    input_text = data['prompt']

    model_name = os.environ.get("CHATBOT_MODEL_NAME")
    # Load model (download on first run and reference local installation for consequent runs)
    hface_auth_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # Load model,tokenizer and convo history
    model, tokenizer, conversation_history = load_model_tokenizer_and_clean_convo_hist(
        model_name=model_name,
        token=hface_auth_token
    )

    # Create conversation history string
    history = "\n".join(conversation_history)

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs, max_length=os.environ.get("MODEL_MAX_TOKEN"))  # max_length will acuse model to crash at some point as history grows

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    load_dotenv()
    app.run(
        debug=True,
        host=os.environ.get("FLASK_SERVER_NAME"),
        port=os.environ.get("FLASK_SERVER_PORT")
    )