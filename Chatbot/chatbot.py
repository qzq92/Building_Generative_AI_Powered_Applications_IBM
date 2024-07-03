from dotenv import load_dotenv
from utils import load_model_tokenizer_and_clean_convo_hist, get_response_for_input
import os

"""
Script which enables chatbot-human interaction via terminal/command prompt application powered by a Huggingface's Chatbot Model with conversation history stored on the fly.
"""

if __name__ == "__main__":
    load_dotenv()

    model_name = os.environ.get(
        "CHATBOT_MODEL_NAME",
         default="facebook/blenderbot-400M-distill"
    )
    # Load model (download on first run and reference local installation for consequent runs)
    hface_auth_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    if not hface_auth_token or hface_auth_token == "":
        raise ValueError("HuggingFace Authentication Token not found.")
    # Load model,tokenizer and convo history
    model, tokenizer, conversation_history = load_model_tokenizer_and_clean_convo_hist(
        model_name=model_name, token=hface_auth_token
    )

    # Conversational loop prompted by > with tracking
    while True:
        history_string = "\n".join(conversation_history)

        # Seek user input prompt
        input_text = input("> ")

        # Pass into a function containing LLM model/tokenizer, history, and input to retrieve LLM completion.
        response = get_response_for_input(
            tokenizer = tokenizer,
            model = model,
            history = history_string,
            input_text = input_text)
        
        # Append to conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)
        print(conversation_history)
        print(response)