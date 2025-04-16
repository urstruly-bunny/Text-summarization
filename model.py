# model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str = None):
        """
        Load tokenizer and model for summarization.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        

    def summarize(self, text: str, max_length: int = 60, min_length: int = 20) -> str:
        """
        Generate a summary for the input text.
        """
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024  # max tokens for encoder input
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2  # helps reduce repetition
        )
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output

if __name__ == "__main__":
    sample_text = (
        "Oliver: Hey, did you talk to the client today?\n"
        "Sophie: Yes, they said they’re fine with the proposal but asked if we can move the deadline.\n"
        "Oliver: Move it how far?\n"
        "Sophie: About a week later. They need more time on their end.\n"
        "Oliver: Hmm, that puts us in a tight spot for the next sprint.\n"
        "Sophie: I know. I told them we’ll try but can’t guarantee.\n"
        "Oliver: Okay, we’ll discuss with the dev team tomorrow.\n"
        "Sophie: Cool. I’ll summarize this in the client notes.\n"
        "Oliver: Thanks, appreciate it!"
    )

    summarizer = Summarizer()
    summary = summarizer.summarize(sample_text)
    print("Summary:", summary)
