from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv()

    model_name = os.environ.get("CHATBOT_MODEL_NAME")
    # Load model (download on first run and reference local installation for consequent runs)
    hface_auth_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # Define model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path = "facebook/blenderbot-400M-distill", token = hface_auth_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="facebook/blenderbot-400M-distill", token=hface_auth_token
    )

    #List to store conversation history
    conversation_history = []
    
    while True:
        history_string = "\n".join(conversation_history)

        input_text = input("> ")
        # Encode history to tokens
        inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

        
        # Gnerate output and decode
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Append to conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)
        #print(conversation_history)
        #print(response)