from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig, pipeline
from langchain.prompts import PromptTemplate
from typing import Tuple
import os
import torch
import base64
import gradio as gr

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def prepare_llama_prompt_and_llm() -> Tuple[PromptTemplate, AutoModelForCausalLM]:
    """Function which instantiates prompt template and LLM model based on Meta's Llama-2-7b-chat-hf Model

    Returns:
        Tuple[PromptTemplate, AutoModelForCausalLM]: Tuple containing PromptTemplate to dictate behavior of model concerned and AutoModelForCausalLM containing pretrained model concerned.
    """

    # Define template with input variables
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


def transcribe_audio_with_distill_stt(audio_binary: bytes) -> str:
    """Function which performs audio transcription using distil-whisper/distil-large-v3 model through Huggingface pipeline construct.
    
    Args:
        audio_binary (bytes): Audio data in raw bytes.

    Returns:
        str: Transcription of audio file.
    """

    stt_model = "distil-whisper/distil-large-v3"

    # Initialize th speech recognition pipeline
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

def generate_transcript_and_summary(audio_filepath: str) -> Tuple[str,str]:
    """Main function which summarised audio content obtained from a input audio_filepath.

    Args:
        audio_filepath (str): Filepath to provided audio file.

    Returns:
        Tuple[str,str]: Tuple containing summarised audio content based on llama model guided by prompts and the transcribed audio
    """
    with open(audio_filepath, "rb") as binary_file:
        # Read the whole file at once
        data = binary_file.read()
        audio_binary = base64.b64encode(data).decode('utf-8')

    transcription = transcribe_audio_with_distill_stt(audio_binary=audio_binary)

    # Get prompt and llm objects
    prompt, llm = prepare_llama_prompt_and_llm()
    

    # Generate summary
    llm_chain = prompt | llm
    summary_result = llm_chain.invoke({"context": transcription})

    return summary_result, transcription


def start_gradio_interface(host:str, port:int):
    """Function which instantiates a Gradio application on specified host and port information.

    Args:
        host (str): Specified server host which Gradio is to run.
        port (int): Specified server port which Gradio is to run.
    """
    demo = gr.Blocks(
        title="OpenAI Transcription with Gradio",
        theme="NoCrypt/miku",
    )

    microphone_interface_title = "Click on 'Record' to start recording your speech for transcription."

    # Define textbox for Gradio UI
    transcription_textbox = gr.Textbox(
        max_lines=10,
        placeholder="Audio transcribed",
        show_copy_button=True,
        label="Audio transcription",
        show_label=True,
        type="text"
    ),

    summary_textbox = gr.Textbox(
        max_lines=10,
        placeholder="Transcription Summary",
        show_copy_button=True,
        label="Transcription Summary",
        show_label=True,
        type="text"
    ),


    # Interface for microphone transcription. Ensure that your browser has access to microphone on the device hosting gradio
    mic_transcribe = gr.Interface(
        fn = generate_transcript_and_summary,
        title = microphone_interface_title,
        inputs = gr.Audio(sources="microphone", type="filepath"),
        outputs = [transcription_textbox, summary_textbox],
        allow_flagging="never"
    )

    file_upload_interface_title = "Upload your audio files here (currently limited to 25 MB) Supported file types: mp3, mp4, mpeg, mpga, m4a, wav, and webm)"
    # Interface for file upload
    file_transcribe = gr.Interface(
        fn = generate_transcript_and_summary,
        title = file_upload_interface_title,
        inputs = gr.Audio(sources="upload", type="filepath"),
        outputs = [transcription_textbox, summary_textbox],
        allow_flagging="never"
    )
    with demo:
        # Construct tab interface with above
        gr.TabbedInterface(
            interface_list = [mic_transcribe, file_transcribe],
            tab_names = ["Transcribe From Microphone", "Transcribe Uploaded Audio File"],
            )
    # Launch
    demo.launch(debug=True, server_name=host, server_port=port, share=False)

if __name__ == "__main__":
    load_dotenv()
    start_gradio_interface(
        host=os.environ.get("GRADIO_SERVER_NAME"),
        port=int(os.environ.get("GRADIO_SERVER_PORT"))
    )