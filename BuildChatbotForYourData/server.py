import logging
import os

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import BuildChatbotForYourData.worker as worker

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the index.html template

# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage']  # Extract the user's message from the request
    print('user_message', user_message)

    # Process prompt from input
    bot_response = worker.process_prompt(user_message)  # Process the user's message using the worker module

    # Return the bot's response as JSON
    return jsonify({
        "botResponse": bot_response
    }), 200

# Define the route for processing documents. This should only be called once for each unique file.
@app.route('/process-document', methods=['POST'])
def process_document_route():
    # Check if a file was uploaded, else return 400 as bad request
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "It seems like the file was not uploaded correctly, can you try again. If the problem persists, try using a different file."
        }), 400

    file = request.files['file']  # Extract the uploaded file from the request
    
    if not file.filename.endswith(".pdf"):
        return jsonify({
            "botResponse": "It seems like the file was not a pdf file, can you please check again."
        }), 400
    
    # Only process subsequently if pdf file is uploaded
    file_path = file.filename  # Define the path where the file will be saved
    file.save(file_path)  # Save the file

    # Process document when the file is valid, supported by worker.py function
    worker.process_document(document_path=file_path)

    # Return a success message as JSON
    return jsonify({
        "botResponse": "Thank you for providing your PDF document. I have analyzed it, so now you can ask me any questions regarding it!"
    }), 200

# Run the Flask app
if __name__ == "__main__":
    app.run(
        debug = True,
        host = os.environ.get("FLASK_RUN_HOST"),
        port = int(os.environ.get("FLASK_RUN_PORT"))
    )
