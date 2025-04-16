# evaluator.py

from rouge_score import rouge_scorer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self):
        # Define the scorer without the 'metrics' argument
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate(self, results):
        references = [result["reference_summary"] for result in results]
        generated = [result["generated_summary"] for result in results]

        rouge_scores = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }

        for ref, gen in zip(references, generated):
            score = self.scorer.score(ref, gen)
            rouge_scores["rouge1"].append(score['rouge1'].fmeasure)
            rouge_scores["rouge2"].append(score['rouge2'].fmeasure)
            rouge_scores["rougeL"].append(score['rougeL'].fmeasure)

        avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
        avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
        avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

        logger.info(f"ROUGE-1: {avg_rouge1}")
        logger.info(f"ROUGE-2: {avg_rouge2}")
        logger.info(f"ROUGE-L: {avg_rougeL}")

        return {
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL
        }

if __name__ == "__main__":
    from pipeline import SummarizationPipeline

    pipeline = SummarizationPipeline()
    results = pipeline.run_inference()

    evaluator = Evaluator()
    evaluator.evaluate(results)
