# data_loader.py

from datasets import load_dataset
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAMSumDataLoader:
    def __init__(self, split_ratio: float = 0.1):
        """
        Initializes the SAMSum dataset.
        :param split_ratio: Fraction of training data to reserve for validation
        """
        logger.info("Loading SAMSum dataset...")
        self.dataset = load_dataset("samsum", trust_remote_code=True)


        # Optional: split train set into train/validation
        if split_ratio > 0:
            logger.info(f"Splitting training data with validation ratio = {split_ratio}")
            self.dataset["train"] = self.dataset["train"].train_test_split(test_size=split_ratio, seed=42)
            self.train_data = self.dataset["train"]["train"]
            self.val_data = self.dataset["train"]["test"]
        else:
            self.train_data = self.dataset["train"]
            self.val_data = None

        self.test_data = self.dataset["test"]

    def get_datasets(self) -> Tuple:
        return self.train_data, self.val_data, self.test_data

    def get_sample(self, split: str = "test", index: int = 0) -> Tuple[str, str]:
        """
        Returns one sample (dialogue, summary) from the chosen split.
        """
        data_split = self.dataset[split]
        sample = data_split[index]
        return sample["dialogue"], sample["summary"]
    
    def get_split(self, split: str = "train"):
        """
        Get a specific split: 'train', 'test', or 'validation'
        """
        if split not in self.dataset:
            raise ValueError(f"Invalid split name: {split}")
        return self.dataset[split]


if __name__ == "__main__":
    loader = SAMSumDataLoader()
    train, val, test = loader.get_datasets()
    logger.info(f"Train samples: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

    dialogue, summary = loader.get_sample("test", 0)
    print("Sample Dialogue:\n", dialogue)
    print("Reference Summary:\n", summary)
