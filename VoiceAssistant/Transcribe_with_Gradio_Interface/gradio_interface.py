import gradio as gr
from openai import OpenAI, AuthenticationError
from dotenv import load_dotenv
import os

def transcribe_speech(filepath: str) -> str:
    """Function which calls OpenAI's audio transcriptions to transcribe a given filepath containing audio speech.

    Args:
        filepath (str): File containing audio recordings.

    Returns:
        str: Transcribed speech if API key for OpenAI is authenticated. Else, return error string based on exception encountered.
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        audio_file = open(filepath, "rb")
        transcript = client.audio.transcriptions.create(
            file = audio_file,
            language="en",
            temperature=0.1,
            model = "whisper-1",
        )

        processed_transcript = str(transcript.text).strip()
        return processed_transcript
    
    except AuthenticationError:
        error_str = "Authentication error encountered. Unable to transcribe."
        return error_str
    except TypeError:
        error_str = "Encountered TypeError when transcribing, please check if you have uploaded correct file or if your file is corrupted."
        return error_str
    
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
    # Interface for microphone transcription. Ensure that your browser has access to microphone on the device hosting gradio
    mic_transcribe = gr.Interface(
        fn = transcribe_speech,
        title = microphone_interface_title,
        inputs = gr.Audio(sources="microphone", type="filepath"),
        outputs = gr.Textbox(
            max_lines=10,
            placeholder="Transcription of speech",
            show_copy_button=True,
            label="Transcription",
            show_label=True,
            type="text"),
        allow_flagging="never"
    )

    file_upload_interface_title = "Upload your audio files here (currently limited to 25 MB) Supported file types: mp3, mp4, mpeg, mpga, m4a, wav, and webm)"
    # Interface for file upload
    file_transcribe = gr.Interface(
        fn = transcribe_speech,
        title = file_upload_interface_title,
        inputs = gr.Audio(sources="upload", type="filepath"),
        outputs = gr.Textbox(
            max_lines=10,
            placeholder="Transcription of audio files",
            show_copy_button=True,
            label="Transcription",
            show_label=True,
            type="text"),
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