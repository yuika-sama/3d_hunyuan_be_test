import logging
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

def load_model(model_name: str):
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        logger.info(f"Loaded model: {model_name}")
        return model, processor
    except Exception:
        logger.exception(f"Error loading model: {model_name}")
        raise

def generate_caption(model, processor, image, prompt: str | None = None, max_new_tokens: int = 60) -> str:
    try:
        if prompt:
            inputs = processor(image, prompt, return_tensors="pt")
        else:
            inputs = processor(image, return_tensors="pt")

        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        logger.info(f"Generated caption: {caption}")
        return caption
    except Exception:
        logger.exception("Error during caption generation")
        raise
