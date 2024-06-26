from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_tokenizer_and_clean_convo_hist(model_name:str, token:str):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token
    )
    conversation_history = []

    return model, tokenizer, conversation_history