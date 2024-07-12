from dotenv import load_dotenv
from langchain_core.prompts import  PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, BarkModel
from language_bark_mapping import language_mapping
from typing import Tuple
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
    Query:
    {query} [/INST] </s>
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

def text_to_speech(input_text: str, language_to_translate_to: str) -> Tuple[np.ndarray,int]:
    """Function which decides to use suno/bark model inference based on to perform text to speech generation for supported language.

    Args:
        input_text (str): Input text to be synthesized.
        language_to_translate (str): Language name which message is to be translated.

    Returns:
        Tuple[np.ndarray,int]: Generated speech values in numpy array and corresponding sampling rate based on TTS model used.
    """

    voice_preset = language_mapping[language_to_translate_to]

    default_model = "suno/bark"
    print(f"Running inference with {default_model}")
    
    # Generate model/processor
    model = BarkModel.from_pretrained(
        pretrained_model_name_or_path=default_model,torch_dtype=TORCH_DTYPE
    )
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=default_model, torch_dtype=TORCH_DTYPE
    )
    # performs kernel fusion under the hood. You can gain 20% to 30% in speed with zero performance degradation.
    model = model.to_bettertransformer()
    
    # Offload idle submodels if using CUDA
    if DEVICE == "cuda:0":
        model.enable_cpu_offload()

    # Tokenize and encode th e text prompt
    inputs = processor(
        text=[input_text],
        voice_preset=f"v2/{voice_preset}_speaker_4",
        return_tensors="pt",
    )

    # a mono 24 kHz speech
    speech_array = model.generate(
        **inputs,
    ).cpu().numpy().squeeze()
    sampling_rate= model.generation_config.sample_rate

    return speech_array, sampling_rate


def process_message(user_message: str, language_to_translate_to:str) -> str:
    """Function which conducts message translation processing using 'mistralai/Mixtral-8x7B-Instruct-v0.1' model, supported by the custom prompt format required.

    Args:
        user_message (str): User provided message in string.
        language_to_translate (str): Language name which message is to be translated.

    Returns:
        str: Langchain LLM completion.
    """
    
    prompt_str = f"You are an assistant helping translate sentences from English into {language_to_translate_to}. Translate the Query below."

    # Get template
    template, llm = get_mistral_prompt_and_llm()

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"prompt_str": prompt_str},
    )
    # Chain prompt and llm
    llm_chain = prompt | llm
    
    response_text = llm_chain.invoke(user_message)

    return response_text