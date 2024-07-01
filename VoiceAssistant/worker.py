import json
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline, WhisperProcessor,WhisperForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from datetime import datetime
import requests
import os
import soundfile as sf
import torch

load_dotenv()




DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def get_cmu_arctic_embedding() -> torch.tensor:
    """Function which returns a speaker embeddings from open-source dataset

    Returns:
        torch.tensor: Torch tensor representing speaker embedding.
    """
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    speaker_embeddings = embeddings_dataset[7306]["xvector"]
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    
    return speaker_embeddings

def speech_to_text(audio_binary: list) -> str:
    """The function simply takes audio_binary as the only parameter and then sends it in the body of the HTTP request.

    Args:
        audio_binary (str): Array of values sampled from audio.

    Returns:
        str: Transcribed speech in text.
    """
    stt_model = os.environ.get("HUGGINGFACE_STT_MODEL_NAME", default="openai/whisper-small")

    if not stt_model or stt_model == "":
        stt_model = "openai/whisper-small"

    stt_temperature = float(
    os.environ.get("HUGGINGFACE_STT_MODEL_TEMPERATURE", 0.0)
)
    # Correction mechanism
    if stt_temperature < 0:
        print("Encountered negative temperature value, overriding to 0 instead.")
        stt_temperature = 0.0

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
            temperature=stt_temperature
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
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=stt_model
    )

    # Create huggingface pipeline that is loaded to suitable DEVICE.
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        max_new_tokens=int(
            os.environ.get("HUGGINGFACE_STT_MODEL_MAX_TOKEN", default= "128")
        ),
        device=DEVICE,
        do_sample=True,
        framework="pt",
        temperature=float(
            os.environ.get("HUGGINGFACE_STT_MODEL_TEMPERATURE", "0.0")
        )
    )
    result = pipe(audio_binary)
    return result["text"]

def text_to_speech(input_text: str) -> json:
    """Function which calls TTS model as a service through inference endpoint API to perform text to speech generation.

    Args:
        input_text (str): Input text to be synthesized.

    Returns:
        json: Response in json object.
    """


    # Load pretrained processor,model,vocoder and embeddings
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    speaker_embeddings = get_cmu_arctic_embedding()

    if os.environ.get("TTS_API_CALL_ENABLED") == "1":
        print("Using API calls for Text-to-speech synthesization\n")
        # Set the headers for our HTTP request
        headers = {
            "Authorization": f"Bearer {os.environ.get("HUGGINGFACEHUB_API_TOKEN")}"
        }

        model_id = os.environ.get("HUGGINGFACE_TTS_MODEL_NAME")
        api_end_point = f"https://api-inference.huggingface.co/models/{model_id}"
        try:
            response = requests.post(
                url=api_end_point,
                headers=headers,
                json=input_text,
                timeout=300
            )

            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("Encountered Server error:", errh)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print("Request exception error",err)

        return response.json()

    print("Running inference offline....")
    # Generate processor
    inputs = processor(text=input_text, return_tensors="pt")
    # Include pad_token_id to suppress warning: "Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation." Returns audio and sampling rate
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=speaker_embeddings,
        vocoder=vocoder,
        threshold=0.5
    )

    # Returns dict of audio and sampling rate
    print(speech)
    tts_audio_files_dir = "tts_audio_files"

    # Write audio file after synthesizing
    os.makedirs(tts_audio_files_dir, exist_ok=True)
    print("Writing to audio file after generating speech...")
    datetime_now_fmt = datetime.now().strftime('%Y%m%d_%H%M%S')
    tts_audio_filename =  f"speech_{datetime_now_fmt}.mp3"
    try:
        sf.write(tts_audio_filename, speech.numpy(), samplerate=24000)
    except TypeError:
        print(f"Unable to write audio file as {tts_audio_filename} due to invalid format")

    return speech.numpy()

def openai_process_message(user_message: str) -> str:
    """Function which processes user message input with OpenAI models and generates completion as response.

    Args:
        user_message (str): User provided message in string.

    Returns:
        str: OpenAI LLM model completion string.
    """
    # Set the prompt for OpenAI Api
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."
    # Call the OpenAI Api to process our prompt
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    openai_response = openai_client.chat.completions.create(
        model = os.environ.get("OPENAI_MODEL_NAME"),
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens = int(os.environ.get("OPENAI_MAX_TOKEN", "4000")),
    )

    # Parse the response to get the response message for our prompt
    print("Generating speech with OpenAI...")
    response_text = openai_response.choices[0].message.content
    return response_text