# pipeline.py

from data_loader import SAMSumDataLoader
from model import Summarizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationPipeline:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        logger.info("Initializing pipeline...")
        self.data_loader = SAMSumDataLoader()
        self.model = Summarizer(model_name=model_name)

    def run_inference(self, split="test", num_samples=5):
        logger.info(f"Running summarization on {split} split, {num_samples} samples")

        dataset_split = self.data_loader.get_split(split)
        samples = dataset_split.select(range(num_samples))

        results = []

        for example in samples:
            dialogue = example["dialogue"]
            reference = example["summary"]
            generated = self.model.summarize(dialogue)

            results.append({
                "dialogue": dialogue,
                "reference_summary": reference,
                "generated_summary": generated
            })

        return results

if __name__ == "__main__":
    pipeline = SummarizationPipeline()
    results = pipeline.run_inference()

    for i, result in enumerate(results):
        print(f"\n=== Sample {i+1} ===")
        print("Dialogue:\n", result["dialogue"])
        print("\nReference Summary:\n", result["reference_summary"])
        print("\nGenerated Summary:\n", result["generated_summary"])
