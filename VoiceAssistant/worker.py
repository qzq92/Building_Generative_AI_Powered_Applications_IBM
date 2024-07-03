from openai import OpenAI
from dotenv import load_dotenv
from load_tts_model_processor_vocoder import load_tts_components, get_speaker_embedding
from transformers import pipeline, WhisperProcessor,WhisperForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor
import requests
import os
import numpy as np
import torch

# Load
load_dotenv()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def speech_to_text(audio_binary: list) -> str:
    """The function simply takes audio_binary (a list of values) as the only parameter and then calls the relevant STT model based on configuration for transcribing the audio.

    Args:
        audio_binary (str): Array of values sampled from audio.

    Returns:
        str: Transcribed speech in text.
    """

    default_model = "facebook/s2t-small-librispeech-asr"

    stt_model = os.environ.get("HUGGINGFACE_STT_MODEL_NAME", default=default_model)

    stt_temperature = float(
    os.environ.get("HUGGINGFACE_STT_MODEL_TEMPERATURE", 0.0)
)
    # Correction mechanism
    if stt_temperature < 0:
        print("Encountered negative temperature value, overriding to 0.0 instead.")
        stt_temperature = 0.0

    if "openai/whisper" in stt_model.lower():
        try:    
            model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=stt_model
            )
            processor = WhisperProcessor.from_pretrained(
                pretrained_model_name_or_path=stt_model
            )
        except ValueError:
            print("Invalid OpenAI model chosen. Default to openai/whisper-tiny.en model ")
            stt_model = "openai/whisper-tiny.en"
            model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=stt_model
            )
            processor = WhisperProcessor.from_pretrained(
                pretrained_model_name_or_path=stt_model
            )

        # Process features of audio
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
        temperature=stt_temperature
    )
    result = pipe(audio_binary)
    return result["text"]

def text_to_speech(input_text: str) -> np.ndarray:
    """Function which decides to use TTS model inference endpoint API or conduct offline inference based on environment variable 'HUGGINGFACE_TTS_MODEL_NAME' to perform text to speech generation.

    Args:
        input_text (str): Input text to be synthesized.

    Returns:
        np.ndarray: Speech values in numpy array.
    """
    model_id = os.environ.get("HUGGINGFACE_TTS_MODEL_NAME")
    if os.environ.get("TTS_API_CALL_ENABLED") == "1":
        print("Using API calls for Text-to-speech synthesization\n")
        # Set the headers for our HTTP request
        headers = {
            "Authorization": f"Bearer {os.environ.get("HUGGINGFACEHUB_API_TOKEN")}"
        }
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

        return np.array(response.content)

    print(f"Running offline inference with {model_id}")
    # Generate processor
    model, processor, vocoder = load_tts_components(tts_model_name=model_id)
    # For Microsoft SpeechT5.# Default sampling rate: 16khz
    if vocoder:
        inputs = processor(text=input_text, return_tensors="pt")
        speaker_embeddings = get_speaker_embedding()
        speech_array = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embeddings,
            vocoder=vocoder,
            threshold=0.5
        ).numpy()

    # For BarkModel, which doesnt need vocoder to generate speech waves
    else:
        inputs = processor(
            text=input_text,
            return_tensors="pt",
            voice_preset="v2/en_speaker_6"
        )

        speech_array = model.generate(
            **inputs,
            threshold=0.5
        ).numpy()

    speech_array = np.array(speech_array, dtype=np.float64)
    return speech_array


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