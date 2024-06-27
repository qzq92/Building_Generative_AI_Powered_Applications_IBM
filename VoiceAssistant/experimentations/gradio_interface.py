import gradio as gr
from transformers import pipeline


"""Simple gradio experimentation for transcribing audio inputs.

Returns:
    _type_: _description_
"""

model_id = "facebook/wav2vec2-base-100h"  # update with your model id
pipe = pipeline(task="automatic-speech-recognition", model=model_id)


def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "english",
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]

demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.Textbox(),
)

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.Textbox(),
)

if __name__ == "__main__":
    with demo:
        gr.TabbedInterface(
            [mic_transcribe, file_transcribe],
            ["Transcribe Microphone", "Transcribe Audio File"],
        )

    demo.launch(debug=True)