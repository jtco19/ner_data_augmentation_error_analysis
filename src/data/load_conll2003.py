"""
Script to load and subsample the CoNLL-2003 NER dataset.
"""

import logging
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_conll2003_dataset():
    """
    Load the CoNLL-2003 dataset from Hugging Face.

    Returns:
        dataset: The loaded dataset object with 'train', 'validation', and 'test' splits.
    """
    logger.info("Starting to load CoNLL-2003 dataset from Hugging Face")
    dataset = load_dataset("conll2003")
    logger.info(f"Successfully loaded dataset with splits: {list(dataset.keys())}")
    return dataset


def subsample_dataset(dataset, sample_rate: float):
    """
    Create a subsampled version of the dataset.

    Args:
        dataset: The loaded dataset object with multiple splits
        sample_rate: A float between 0 and 1 indicating the fraction of data to keep.
                    For example, 0.5 keeps 50% of the data, 0.1 keeps 10%.

    Returns:
        A new dataset with subsampled splits. Each split is sampled proportionally.

    Raises:
        ValueError: If sample_rate is not between 0 and 1.
    """
    logger.info(f"Starting subsampling with sample_rate={sample_rate}")

    if not (0 < sample_rate <= 1):
        logger.error(f"Invalid sample_rate: {sample_rate}. Must be between 0 and 1")
        raise ValueError(f"sample_rate must be between 0 and 1, got {sample_rate}")

    subsampled = {}

    for split_name, split_data in dataset.items():
        # Calculate the number of samples to keep
        original_size = len(split_data)
        new_size = max(1, int(original_size * sample_rate))

        # Select random indices
        indices = np.random.choice(original_size, size=new_size, replace=False)

        # Create subsampled split
        subsampled[split_name] = split_data.select(indices)

        logger.info(
            f"Subsampled {split_name}: {original_size} -> {new_size} samples ({sample_rate*100:.1f}%)"
        )
        print(
            f"Subsampled {split_name}: {original_size} -> {new_size} samples ({sample_rate*100:.1f}%)"
        )

    logger.info(f"Successfully completed subsampling")
    return subsampled


if __name__ == "__main__":
    # Load the dataset
    print("Loading CoNLL-2003 dataset...")
    dataset = load_conll2003_dataset()
    print(f"Dataset loaded successfully with splits: {list(dataset.keys())}")
