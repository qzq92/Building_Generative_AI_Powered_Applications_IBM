from transformers import pipeline
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from typing import Tuple
import gradio as gr
import os
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def prepare_llama_prompt_and_llm() -> Tuple[PromptTemplate, AutoModelForCausalLM]:
    """Function which instantiates prompt template and LLM model based on Meta's Llama-2-7b-chat-hf Model

    Returns:
        Tuple[PromptTemplate, AutoModelForCausalLM]: Tuple containing PromptTemplate to dictate behavior of model concerned and AutoModelForCausalLM containing pretrained model concerned.
    """

    # Define template
    template = """
    <s><<SYS>>
    List the key points with details from the context: 
    [INST] The context : {context} [/INST] 
    <</SYS>>
    """

    prompt = PromptTemplate(
        input_variables=["context"],
        template=template
    )

    model_id = "meta-llama/Llama-2-7b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.Tensor.bfloat16
    )

    llm = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        quantization_config=bnb_config if DEVICE == "cuda:0" else None,
        device_map="auto",
    )

    return prompt, llm

# Function to transcribe audio using the OpenAI Whisper model
def transcribe_audio_with_whisper_tiny(audio_filepath: str) -> str:
    """Function which performs audio transcription using OpenAI's whisper-tiny english model through Huggingface pipeline construct.
    
    Args:
        audio_filepath (str): Filepath to audio file

    Returns:
        str: Transcription of audio file.
    """
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        )
    
    # Perform speech recognition on the audio file
    # The `batch_size=8` parameter indicates how many chunks are processed at a time. The result is stored in `prediction` with the key "text" containing the transcribed text
    try:
        result = pipe(audio_filepath, batch_size=8)["text"]
        # Print the transcribed text to the console
    except Exception as e:
        raise(f"Encountered error with {e}")

    return result


def summarise_audio_content(audio_filepath: str) -> str:

    transcription = transcribe_audio_with_whisper_tiny(audio_filepath=audio_filepath)

    # Get prompt and llm objects
    prompt, llm = prepare_llama_prompt_and_llm()

    llm_chain = prompt | llm
    result = llm_chain.invoke({"context": transcription})

    return result

if __name__ == "__main__":
    load_dotenv()
    # Set up Gradio interface
    audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
    output_text = gr.Textbox()  # Text output

    # Create the Gradio interface with the function, inputs, and outputs
    iface = gr.Interface(fn=summarise_audio_content,
                        inputs=audio_input,
                        outputs=output_text,
                        title="Audio Transcription App",
                        description="Upload the audio file")
    # Launch the Gradio app
    iface.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT"))
    )