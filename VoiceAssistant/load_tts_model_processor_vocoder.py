from typing import Tuple, Union
from transformers import  AutoProcessor, BarkModel, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def get_speaker_embedding() -> torch.tensor:
    """Function which loads the CMU ARCTIC dataset speaker embeddings from HuggingFace Hub and returns the xvector representing the speaker embeddings.

    Returns:
        torch.tensor: Torch tensor representing speaker embedding.
    """

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    speaker_embeddings_xvec = embeddings_dataset[7306]["xvector"]
    speaker_embeddings = torch.tensor(speaker_embeddings_xvec).unsqueeze(0)
    
    return speaker_embeddings

def load_tts_components(tts_model_name:str) -> Tuple[
    Union[SpeechT5Processor, AutoProcessor],
    Union[SpeechT5ForTextToSpeech, BarkModel],
    Union[SpeechT5HifiGan , None]
]:
    """Function which returns text-to-speech components involving processor, model and vocoder. Currently supports model of suno/bark and microsoft/speecht5 model.'

    Fallsback to base microsoft/speecht5 if variant model name containing speecht5 substring provided is not found in HuggingFace. Otherwise, fall back to suno/bark-small model as last resort.

    Returns:
        Tuple: Tuple containing pretrained text-to-speech processor, text-to-speech model and text-to-speech vocoder (if applicable). 
    """

    # Default model and vocoder (for speecht5)
    if "speecht5" in tts_model_name.lower():
        vocoder = SpeechT5HifiGan.from_pretrained(
            pretrained_model_name_or_path="microsoft/speecht5_hifigan", torch_dtype=TORCH_DTYPE
        )
        # Load pretrained processor,model,vocoder and embeddings involving speecht5
        try:
            processor = SpeechT5Processor.from_pretrained(
                pretrained_model_name_or_path=tts_model_name, torch_dtype=TORCH_DTYPE)
            
            model = SpeechT5ForTextToSpeech.from_pretrained(
                pretrained_model_name_or_path=tts_model_name, torch_dtype=TORCH_DTYPE)

        except (ValueError, MemoryError):
            print("Defaulting to Microsoft's speecht5 model")
            t5_model_name = "microsoft/speecht5"

            processor = SpeechT5Processor.from_pretrained(
                pretrained_model_name_or_path=t5_model_name, torch_dtype=TORCH_DTYPE
            )
            model = SpeechT5ForTextToSpeech.from_pretrained(
                pretrained_model_name_or_path=t5_model_name, torch_dtype=TORCH_DTYPE
            )
    # Load pretrained bark processor, model
    elif "suno/bark" in tts_model_name:
        vocoder = None
        try:
            model = BarkModel.from_pretrained(
                pretrained_model_name_or_path=tts_model_name, torch_dtype=TORCH_DTYPE
            )
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=tts_model_name, torch_dtype=TORCH_DTYPE
            )

        except (ValueError, MemoryError):
            print(f"Defaulting to {default_model} model")
            model = BarkModel.from_pretrained(
                pretrained_model_name_or_path=default_model,torch_dtype=TORCH_DTYPE
            )
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=default_model, torch_dtype=TORCH_DTYPE
            )
        finally:
            # performs kernel fusion under the hood. You can gain 20% to 30% in speed with zero performance degradation.
            model = model.to_bettertransformer()
            # Offload idle submodels if using CUDA
            if DEVICE == "cuda:0":
                model.enable_cpu_offload()
    # Fallback case
    else:
        default_model = "suno/bark"
        print(f"Defaulting to {default_model} model as entered model is unsupoorted.")
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
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=default_model,
            torch_dtype=TORCH_DTYPE
        )
        vocoder = None

    # Load to necessary device
    model =  model.to(DEVICE)
    return model, processor, vocoder