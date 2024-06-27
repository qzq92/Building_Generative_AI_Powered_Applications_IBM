from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple

def load_model_tokenizer_and_clean_convo_hist(model_name:str, token:str) -> Tuple(AutoModelForSeq2SeqLM, AutoTokenizer, list):
    """Function which loads transformers library AutoModelForSeq2SeqLM, AutoTokenizer based on model and input token and returns them together with an empty list of conversation history.

    Args:
        model_name (str): Valid model name found in HuggingFaceHub.
        token (str): HuggingFaceHub API token.

    Returns:
        Tuple: Containing AutoModelForSeq2SeqLM, AutoTokenizer, list
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path = model_name,
        token = token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = model_name,
        token = token
    )
    conversation_history = []

    return model, tokenizer, conversation_history