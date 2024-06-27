from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple
import os 


def load_model_tokenizer_and_clean_convo_hist(model_name:str, token:str) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, list]:
    """Function which loads transformers library AutoModelForSeq2SeqLM, AutoTokenizer based on model and input token and returns them together with an empty list of conversation history.

    Args:
        model_name (str): Valid model name found in HuggingFaceHub.
        token (str): HuggingFaceHub API token.

    Returns:
        Tuple: Containing AutoModelForSeq2SeqLM, AutoTokenizer, list
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path = model_name,
        token = token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = model_name,
        token = token
    )
    conversation_history = []

    return model, tokenizer, conversation_history

def get_response_for_input(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    history: str,
    input_text: str
    ) -> str:
    """Function which returns a LLM model generated output by feeding in conversation history, tokenized input text.

    Args:
        tokenizer (AutoTokenizer): Huggingface tokenizer based on a specified model name that is also used for model.
        model (AutoModelForSeq2SeqLM): A HuggingFaceHub model belonging to Seq2Seq family based on a specified model name that is also used for tokenizer.
        history (str): Context of conversation.
        input_text (str): Input text to be tokenized.

    Returns:
        str: Generated LLM completion.
    """

    # Tokenize the input text and history with encode_plus method from tokenizer
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(
        **inputs,
        temperature = 0,
        do_sample=True,
        top_k = 10, 
        early_stopping = True,
        max_length = int(os.environ.get("MODEL_MAX_LENGTH"))
    )  # max_length will cause model to crash at some point as history grows

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return response