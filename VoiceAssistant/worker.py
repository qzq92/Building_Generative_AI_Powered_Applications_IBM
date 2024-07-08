from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError, OpenAI
from VoiceAssistant.load_tts_model_processor_vocoder import load_tts_components, get_speaker_embedding
from langchain_core.prompts import  PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from typing import Tuple
import requests
import os
import numpy as np
import torch
import numpy as np

# Load env
load_dotenv()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def get_mistral_prompt_and_llm()-> Tuple[str, HuggingFaceEndpoint]:
    """Function which returns preconstructed prompt template required by 
    mistralai/Mixtral-8x7B-Instruct-v0.1 LLM model and the LLM model itself.

    Returns:
        str: Transcribed speech in text.
    """
    # Prompt template required for mixtral model
    template = """<s>[INST] {prompt_str}
    Answer the question below:
    {question} [/INST] </s>
    """

    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=512
    )

    return template, llm

def speech_to_text(audio_binary: bytes) -> str:
    """The function simply takes audio_binary (a list of values) as the only parameter and then calls the relevant STT model based on configuration for transcribing the audio.

    Args:
        audio_binary (bytes): Audio data in raw bytes.

    Returns:
        str: Transcribed speech in text.
    """

    stt_model = os.environ.get(
        "HUGGINGFACE_STT_MODEL_NAME",
        default="distil-whisper/distil-large-v3"
    )

    print(audio_binary)

    # Convert audio sound bytes to numpy array.
    # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
    # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
    audio_binary_buffer = np.frombuffer(audio_binary, dtype=np.int16).astype(np.float32) / 32768.0

    print(audio_binary_buffer)
    
    # Assume other models that is not from OpenAI
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        pretrained_model_name_or_path=stt_model,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=stt_model
    )

    # Create huggingface pipeline that is loaded to suitable DEVICE config.
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        device=DEVICE,
    )
    print("Transcribing...")
    result = pipe(audio_binary)
    return result["text"]

def text_to_speech(input_text: str) -> Tuple[np.ndarray,int]:
    """Function which decides to use TTS model inference endpoint API or conduct offline inference based on environment variable 'HUGGINGFACE_TTS_MODEL_NAME' to perform text to speech generation.

    Args:
        input_text (str): Input text to be synthesized.

    Returns:
        Tuple[np.ndarray,int]: Generated speech values in numpy array and corresponding sampling rate based on TTS model used.
    """
    model_id = os.environ.get(
        "HUGGINGFACE_TTS_MODEL_NAME",
        default="suno/bark"
    )
    if os.environ.get("TTS_API_CALL_ENABLED"):
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

    print(f"Running inference with {model_id}")
    # Generate processor
    model, processor, vocoder = load_tts_components(tts_model_name=model_id)
    # For Microsoft SpeechT5.# Default sampling rate: 16khz. Requires speaker embedding for speech pronunciation tones
    if vocoder:
        inputs = processor(text=input_text, return_tensors="pt")
        speaker_embeddings = get_speaker_embedding()
        speech_array = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embeddings,
            vocoder=vocoder,
            threshold=0.5
        ).numpy()
        sampling_rate = 16000
    # For bark models, which doesnt need vocoder to generate speech waves
    else:
        # Tokenize and encode th e text prompt
        inputs = processor(
            text=[input_text],
            voice_preset="v2/en_speaker_6",
            return_tensors="pt",
        )
        # a mono 24 kHz speech
        speech_array = model.generate(
            **inputs,
        ).cpu().numpy().squeeze()
        sampling_rate= model.generation_config.sample_rate

    return speech_array, sampling_rate


def process_message(user_message: str) -> str:
    """Function which uses OpenAI Chat Model to process message if the event that 'OPENAPI_CHATMODEL_API_CALL_ENABLED' environment is set. Otherwise, default prompt and LLM model based on 'mistralai/Mixtral-8x7B-Instruct-v0.1' would be used instead for message processing.

    Args:
        user_message (str): User provided message in string.

    Returns:
        str: Langchain LLM completion.
    """
    
    # Call the OpenAI Api to process our prompt
    is_openai_enabled = os.environ.get("OPENAPI_CHATMODEL_API_CALL_ENABLED", default="")

    prompt_str = "Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations at the end."


    if is_openai_enabled:
        print(f"Using OpenAI model for input: {user_message}")
        openai_model = os.environ.get("OPENAI_MODEL_NAME")
        try:
            # Call the OpenAI Api to process our prompt
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            openai_response = openai_client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": prompt_str},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=4000
            )
            print("openai response:", openai_response)
            # Parse the response to get the response message for our prompt
            response_text = openai_response.choices[0].message.content
            return response_text
        except (AuthenticationError, RateLimitError, ValueError) as err:
            print(f"Encountered {err}")
            template, llm = get_mistral_prompt_and_llm()

    # Use default
    template, llm = get_mistral_prompt_and_llm()

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"prompt_str": prompt_str},
    )
    # Chain prompt and llm
    llm_chain = prompt | llm
    try:
        response_text = llm_chain.invoke(user_message)
    except RateLimitError:
        print("Encountered Rate Limit Error")
    
    return response_text