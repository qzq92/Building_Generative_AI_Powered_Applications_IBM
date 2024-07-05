from dotenv import load_dotenv
from openai import AuthenticationError, RateLimitError
from langchain_openai import ChatOpenAI
from VoiceAssistant.load_tts_model_processor_vocoder import load_tts_components, get_speaker_embedding
from langchain_core.prompts import  PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline, WhisperProcessor,WhisperForConditionalGeneration, AutoModelForSpeechSeq2Seq, AutoProcessor
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


def get_mistral_prompt_and_llm()-> Tuple(str, HuggingFaceEndpoint):
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
    default_model = "facebook/s2t-small-librispeech-asr"

    stt_model = os.environ.get("HUGGINGFACE_STT_MODEL_NAME", default=default_model)

    stt_temperature = float(
        os.environ.get("HUGGINGFACE_STT_MODEL_TEMPERATURE", default=0.1)
    )
    # Correction mechanism
    if stt_temperature <= 0:
        print("Encountered non-positive temperature value, overriding to 0.1 instead.")
        stt_temperature = 0.1

    print(audio_binary)
    print()
    
    # Convert audio sound bytes to numpy array
    audio_binary_buffer = np.frombuffer(audio_binary, dtype='int16')
    print(audio_binary_buffer)
    if "openai/whisper" in stt_model.lower():
        print(f"Using openai whisper model: {stt_model}")
        try:    
            model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=stt_model
            )
            processor = WhisperProcessor.from_pretrained(
                pretrained_model_name_or_path=stt_model,
                language="en",
                task="transcribe"
            )
        except ValueError:
            print("Invalid OpenAI model chosen. Default to openai/whisper-tiny.en model ")
            stt_model = "openai/whisper-tiny.en"
            model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=stt_model
            )
            processor = WhisperProcessor.from_pretrained(
                pretrained_model_name_or_path=stt_model,
                language="en",
                task="transcribe"
            )

        # pre-process to get the input features
        input_features = processor(
            audio_binary_buffer,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        

        # Generate token ids and decode them (returns a list). Configuration would be impactful for long form transcription
        predicted_ids = model.generate(
            input_features=input_features,
            language="en",
            task="transcribe",
            do_sample=True,
            temperature=stt_temperature
        )

        # post-process token ids to text
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )

        print("Transcribing...")
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
    print("Transcribing...")
    result = pipe(audio_binary)
    return result["text"]

def text_to_speech(input_text: str) -> np.ndarray:
    """Function which decides to use TTS model inference endpoint API or conduct offline inference based on environment variable 'HUGGINGFACE_TTS_MODEL_NAME' to perform text to speech generation.

    Args:
        input_text (str): Input text to be synthesized.

    Returns:
        np.ndarray: Speech values in numpy array.
    """
    model_id = os.environ.get("HUGGINGFACE_TTS_MODEL_NAME", default="")
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
            text=[input_text],
            return_tensors="pt",
        )
        # a mono 24 kHz speech
        speech_array = model.generate(
            **inputs,
            do_sample=True
        ).numpy()

    speech_array = np.array(speech_array, dtype=np.float64)
    return speech_array


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
        # Prompt template required for mixtral model
        template = """{prompt_str}

        Question: {question}
        Helpful Answer:
        """
        # Use ChatOpenAI
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=os.environ.get("OPENAI_API_KEY"),
                max_retries=0
            )
        except (AuthenticationError, RateLimitError) as err:
            print(f"Encountered Error {err}")
            template, llm = get_mistral_prompt_and_llm()

        # # Call the OpenAI Api to process our prompt
        # openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
        # openai_response = openai_client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": prompt},
        #         {"role": "user", "content": user_message}
        #     ],
        #     max_tokens=4000
        # )
        # print("openai response:", openai_response)
        # # Parse the response to get the response message for our prompt
        # response_text = openai_response.choices[0].message.content
        # return response_text

    # Use default
    else:
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