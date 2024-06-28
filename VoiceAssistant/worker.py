from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline, WhisperProcessor,WhisperForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor
from typing import Any
from datetime import datetime
import requests
import os
import soundfile as sf
import torch
load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def speech_to_text(audio_binary: list) -> str:
    """The function simply takes audio_binary as the only parameter and then sends it in the body of the HTTP request.

    Args:
        audio_binary (str): Array of values sampled from audio.

    Returns:
        str: Transcribed speech in text.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    stt_model = os.environ.get("STT_MODEL_NAME")

    if not stt_model or stt_model == "":
        stt_model = "openai/whisper-small"
    
    if stt_model.startswith("openai/whisper"):
        model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=stt_model
        )
        processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path=stt_model
        )
        input_features = processor(
            audio_binary, sampling_rate=44100, return_tensors="pt"
        ).input_features
        
        # Generate token ids and decode them (returns a list). Configuration would be impactful for long form transcription
        predicted_ids = model.generate(
            input_features=input_features,
            language="en",
            task="transcribe",
            do_sample=True,
            temperature=float(os.environ.get("MODEL_STT_TEMPERATURE"))
        )
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )
        print(transcription)

        return transcription[0]
    
    # Assume other models that is not from OpenAI
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        pretrained_model_name_or_path=stt_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=stt_model
    )

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        max_new_tokens=int(os.environ.get("MODEL_STT_MAX_TOKEN")),
        device=device,
        do_sample=True,
        temperature=float(os.environ.get("MODEL_STT_TEMPERATURE"))
    )

    result = pipe(audio_binary)
    return result["text"]

def text_to_speech(input_text: str, api_call: bool = True) -> Any:
    """Function which calls TTS model as a service through inference endpoint API to perform text to speech generation.

    Args:
        input_text (str): Input text to be synthesized.
        api_call (bool): Boolean state to control whether to do inference via API call directly.
    """
    if api_call:
        # Set the headers for our HTTP request
        headers = {
            "Authorization": f"Bearer {os.environ.get("HUGGINGFACEHUB_API_TOKEN")}"
        }

        model_name = os.environ.get("HUGGINGFACE_TTS_MODEL_NAME")
        inference_base_api = "https://api-inference.huggingface.co/models/"
        api_end_point = inference_base_api + model_name

        # Set the body of our HTTP request
        request_payload = {
            "text_inputs": input_text,
        }

        response = requests.post(
            url=api_end_point,
            headers=headers,
            json=request_payload,
            timeout=15
        )
        print('text to speech response:', response)
        return response.content

    # If not api call
    synthesizer = pipeline(
        task="text-to-speech",
        model=os.environ.get("HUGGINGFACE_TTS_MODEL_NAME")
    )

    speech = synthesizer(inputs=input_text)

    tts_audio_files_dir = "tts_audio_files"
    os.makedirs(tts_audio_files_dir, exist_ok=True)

    # Write audio file after synthesizing
    datetime_now_fmt = datetime.now().strftime('%Y%m%d_%H%M%S')
    tts_audio_filename =  f"speech_{datetime_now_fmt}"
    sf.write(tts_audio_filename, speech["audio"], samplerate=speech["sampling_rate"])

    response = synthesizer(inputs=input_text)
    print(response)
    print(type(response))
    return response

def openai_process_message(user_message: str) -> str:
    """Function which processes user message with OpenAI models and generates completion as response.

    Args:
        user_message (str): User provided message in string.

    Returns:
        str: OpenAI LLM model completion string.
    """
    # Set the prompt for OpenAI Api
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."
    # Call the OpenAI Api to process our prompt
    openai_response = openai_client.chat.completions.create(
        model = os.environ.get("OPENAI_MODEL_NAME"),
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens = os.environ.get("OPENAI_MAX_TOKEN")
    )
    print("openai response:", openai_response)
    # Parse the response to get the response message for our prompt
    response_text = openai_response.choices[0].message.content
    return response_text