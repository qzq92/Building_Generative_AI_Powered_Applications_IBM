import gradio as gr
from openai import OpenAI, AuthenticationError
from dotenv import load_dotenv
import os

def transcribe_speech(filepath: str) -> str:
    """Function which calls OpenAI's aduio transcriptions to transcribe a given filepath containing audio speech.

    Args:
        filepath (str): File containing audio recordings.

    Returns:
        str: Transcribed speech if API key for OpenAI is authenticated. Else, return error string.
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        audio_file = open(filepath, "rb")
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        return transcript.words
    except AuthenticationError:
        error_str = "Authentication error encountered. Unable to transcribe."
        return error_str
    
def start_gradio_interface(host:str, port:int):
    """Function which instantiates a Gradio application on specified host and port information.

    Args:
        host (str): Specified server host which Gradio is to run.
        port (int): Specified server port which Gradio is to run.
    """
    demo = gr.Blocks(
        title="OpenAI Transcription with Gradio",
        theme="NoCrypt/miku"
    )

    # Interface for microphone transcription. Ensure that your browser has access to microphone on the device hosting gradio
    mic_transcribe = gr.Interface(
        fn=transcribe_speech,
        inputs=gr.Audio(sources="microphone", type="filepath"),
        outputs=gr.Textbox(),
    )

    # Interface for file upload
    file_transcribe = gr.Interface(
        fn=transcribe_speech,
        inputs=gr.Audio(sources="upload", type="filepath"),
        outputs=gr.Textbox(),
    )
    with demo:
        gr.TabbedInterface(
            [mic_transcribe, file_transcribe],
            ["Transcribe Microphone", "Transcribe Audio File"],
        )

    demo.launch(debug=True, server_name=host, server_port=port, share=False)

if __name__ == "__main__":
    load_dotenv()
    start_gradio_interface(
        host=os.environ.get("GRADIO_SERVER_NAME"),
        port=int(os.environ.get("GRADIO_SERVER_PORT"))
    )