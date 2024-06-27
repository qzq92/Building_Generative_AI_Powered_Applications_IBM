from dotenv import load_dotenv
from utils import load_model_tokenizer_and_clean_convo_hist
import os

"""
Script which enables chatbot-human interaction via terminal/command prompt application powered by a Huggingface's Chatbot Model with conversation history stored on the fly.
"""

if __name__ == "__main__":
    load_dotenv()

    model_name = os.environ.get("CHATBOT_MODEL_NAME")
    # Load model (download on first run and reference local installation for consequent runs)
    hface_auth_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # Load model,tokenizer and convo history
    model, tokenizer, conversation_history = load_model_tokenizer_and_clean_convo_hist(
        model_name=model_name, token=hface_auth_token
    )

    while True:
        history_string = "\n".join(conversation_history)

        # Seek user input prompt
        input_text = input("> ")
        # Encode history to tokens
        inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

        
        # Gnerate output and decode
        outputs = model.generate(
            **inputs,
            max_length=int(os.environ.get("MODEL_MAX_TOKEN")))
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Append to conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)
        print(conversation_history)
        print(response)