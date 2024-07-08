from typing import Tuple
from dotenv import load_dotenv
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor, BitsAndBytesConfig, pipeline
from langchain.prompts import PromptTemplate

import os
import torch
import gradio as gr

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
FILE_SIZE_LIMIT_MB = 25

def prepare_llama_prompt_and_llm() -> Tuple[PromptTemplate, AutoModelForCausalLM, AutoTokenizer]:
    """Function which instantiates prompt template, LLM model and Tokenizer objects based on Meta's Llama-2-7b-chat-hf Model

    Returns:
        Tuple[PromptTemplate, AutoModelForCausalLM]: Tuple containing PromptTemplate to dictate behavior of model concerned and AutoModelForCausalLM containing pretrained model concerned.
    """

    # Define template with input variables
    template = """
    <s>[INST] 
    List 3 key points with details from the context below separated by new line without any indentation: 
    
    Context: {context}
    [/INST] 
    """

    prompt = PromptTemplate(
        input_variables=["context"],
        template=template
    )

    model_id = "meta-llama/Llama-2-7b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4', #for weights initialized using a normal distribution.
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 #Changing the Compute Data Type
    )
    
    # Pretrained llm
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        quantization_config=bnb_config if DEVICE == "cuda:0" else None,
        device_map="auto",
    )
    # Tokenizer for same model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False

    # Task pipeline
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        max_length=512,
        truncation=True,
        do_sample=True,
        return_full_text=False
    )

    llm = HuggingFacePipeline(
        pipeline= hf_pipeline,
        model_kwargs = {
            "temperature": 0,
        })
    
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

def generate_transcript_and_summary(audio_data: bytes) -> Tuple[str,str]:
    """Main function which summarised audio content obtained from a input audio_filepath.

    Args:
        audio_filepath (str): Filepath to provided audio file.

    Returns:
        Tuple[str,str]: Tuple containing summarised audio content based on llama model guided by prompts and the transcribed audio.
    """
    file_size_mb = os.stat(audio_data).st_size / (1024 * 1024)
    if file_size_mb > FILE_SIZE_LIMIT_MB:
        print(f"Max file size exceeded {FILE_SIZE_LIMIT_MB}")
        raise gr.Error(
            f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_SIZE_LIMIT_MB}MB."
        )
    transcription = transcribe_audio_with_distill_stt(audio_binary=audio_data)
    
    print()
    print("Generated transcription")
    print(transcription)

    # Get prompt and llm objects
    prompt, llm = prepare_llama_prompt_and_llm()

    # Generate summary
    llm_chain = prompt | llm
    summary = llm_chain.invoke({"context": transcription})

    print("Generated summary:")
    print(summary)

    return transcription, summary


def start_gradio_interface(host:str, port:int):
    """Function which instantiates a Gradio application on specified host and port information.

    Args:
        host (str): Specified server host which Gradio is to run.
        port (int): Specified server port which Gradio is to run.
    """
    demo = gr.Blocks(
        title="Audio transcription and Summarizer tool",
        theme="NoCrypt/miku",
    )

    microphone_interface_title = "Click on 'Record' to start recording your speech for transcription."

    # For microphone transcription and summary
    mic_transcription_textbox = gr.Textbox(
        max_lines=5,
        placeholder="",
        show_copy_button=True,
        label="Microphone audio transcription",
        show_label=True,
        type="text"
    )

    mic_summary_textbox = gr.Textbox(
        max_lines=5,
        placeholder="",
        show_copy_button=True,
        label="Transcription summary of microphone input",
        show_label=True,
        type="text"
    )

    # Interface for microphone transcription. Ensure that your browser has access to microphone on the device hosting gradio
    mic_transcribe = gr.Interface(
        fn = generate_transcript_and_summary,
        title = microphone_interface_title,
        inputs = gr.Audio(sources="microphone", type="filepath"),
        outputs = [mic_transcription_textbox, mic_summary_textbox],
        allow_flagging="never"
    )

    # For Audiofile transcription and summary
    audiofile_transcription_textbox = gr.Textbox(
        max_lines=5,
        placeholder="",
        show_copy_button=True,
        label="Audio file transcription",
        show_label=True,
        type="text"
    )

    audiofile_summary_textbox = gr.Textbox(
        max_lines=5,
        placeholder="",
        show_copy_button=True,
        label="Transcription summary of audio file",
        show_label=True,
        type="text"
    )


    file_upload_interface_title = "Upload your audio files here (currently limited to 25 MB)"

    # Interface for file upload
    file_transcribe = gr.Interface(
        fn = generate_transcript_and_summary,
        title = file_upload_interface_title,
        inputs = gr.Audio(sources="upload", type="filepath"),
        outputs = [audiofile_transcription_textbox, audiofile_summary_textbox],
        allow_flagging="never"
    )
    with demo:
        # Construct tab interface with above
        gr.TabbedInterface(
            interface_list = [mic_transcribe, file_transcribe],
            tab_names = ["Transcribe From Microphone", "Transcribe Uploaded Audio File"],
        )
    # Launch
    print("Launching gradio...")
    demo.launch(debug=True, server_name=host, server_port=port, share=False)

if __name__ == "__main__":
    load_dotenv()
    start_gradio_interface(
        host=os.environ.get("GRADIO_SERVER_NAME"),
        port=int(os.environ.get("GRADIO_SERVER_PORT"))
    )