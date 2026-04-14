"""
Comprehensive test suite for load_conll2003.py using pytest.
"""

import sys
import os
from pathlib import Path

# Add the data directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

import pytest
import numpy as np
from datasets import Dataset, DatasetDict
from load_conll2003 import load_conll2003_dataset, subsample_dataset
import logging

# Get logger for tests (pytest will handle configuration)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_dataset():
    """
    Create a mock CoNLL-2003-like dataset for testing.

    Returns:
        A DatasetDict with train, validation, and test splits.
    """
    # Define label names
    label_names = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]

    def create_split(num_samples: int, seed: int):
        """Helper function to create a dataset split."""
        np.random.seed(seed)
        tokens_list = []
        tags_list = []
        pos_tags_list = []
        chunk_tags_list = []

        for _ in range(num_samples):
            # Random number of tokens between 5 and 20
            num_tokens = np.random.randint(5, 21)
            tokens = [f"token_{i}" for i in range(num_tokens)]
            tags = [np.random.randint(0, len(label_names)) for _ in range(num_tokens)]
            pos_tags = [
                np.random.randint(0, 45) for _ in range(num_tokens)
            ]  # Standard POS tags
            chunk_tags = [
                np.random.randint(0, 22) for _ in range(num_tokens)
            ]  # Standard chunk tags

            tokens_list.append(tokens)
            tags_list.append(tags)
            pos_tags_list.append(pos_tags)
            chunk_tags_list.append(chunk_tags)

        # Create dataset
        data = {
            "tokens": tokens_list,
            "ner_tags": tags_list,
            "pos_tags": pos_tags_list,
            "chunk_tags": chunk_tags_list,
        }

        dataset = Dataset.from_dict(data)

        # Add features with proper formatting
        from datasets import Features, Sequence, Value, ClassLabel

        features = Features(
            {
                "tokens": Sequence(Value("string")),
                "ner_tags": Sequence(ClassLabel(names=label_names)),
                "pos_tags": Sequence(Value("int32")),
                "chunk_tags": Sequence(Value("int32")),
            }
        )

        dataset = dataset.cast(features)
        return dataset

    # Create splits
    train_split = create_split(100, seed=42)
    validation_split = create_split(10, seed=43)
    test_split = create_split(10, seed=44)

    dataset_dict = DatasetDict(
        {
            "train": train_split,
            "validation": validation_split,
            "test": test_split,
        }
    )

    return dataset_dict


class TestLoadConll2003Dataset:
    """Tests for load_conll2003_dataset function."""

    def test_load_dataset_returns_dataset_dict(self):
        """Test that load_conll2003_dataset returns a DatasetDict."""
        dataset = load_conll2003_dataset()
        assert isinstance(dataset, DatasetDict), "Should return a DatasetDict"

    def test_dataset_has_required_splits(self):
        """Test that dataset has train, validation, and test splits."""
        dataset = load_conll2003_dataset()
        required_splits = {"train", "validation", "test"}
        assert (
            set(dataset.keys()) == required_splits
        ), f"Dataset should have splits: {required_splits}"

    def test_dataset_has_required_columns(self):
        """Test that each split has 'tokens' and 'ner_tags' columns."""
        dataset = load_conll2003_dataset()
        required_columns = {"tokens", "ner_tags"}

        for split_name, split_data in dataset.items():
            assert required_columns.issubset(
                set(split_data.column_names)
            ), f"Split '{split_name}' should have columns: {required_columns}"

    def test_dataset_non_empty(self):
        """Test that each split has at least some samples."""
        dataset = load_conll2003_dataset()
        for split_name, split_data in dataset.items():
            assert len(split_data) > 0, f"Split '{split_name}' should not be empty"


class TestSubsampleDataset:
    """Tests for subsample_dataset function."""

    def test_subsample_with_valid_rate(self, sample_dataset):
        """Test subsampling with valid sample rates."""
        sample_rate = 0.5
        subsampled = subsample_dataset(sample_dataset, sample_rate)

        assert isinstance(subsampled, dict), "Should return a dictionary"

        # Check that all splits are present
        assert set(subsampled.keys()) == set(
            sample_dataset.keys()
        ), "Subsampled dataset should have same splits as original"

        # Check that sizes are approximately correct
        for split_name in sample_dataset.keys():
            original_size = len(sample_dataset[split_name])
            subsampled_size = len(subsampled[split_name])
            expected_size = int(original_size * sample_rate)

            assert (
                subsampled_size == expected_size
            ), f"Split '{split_name}': expected {expected_size} samples, got {subsampled_size}"

    def test_subsample_with_low_rate(self, sample_dataset):
        """Test subsampling with very low sample rate (20%)."""
        sample_rate = 0.2
        subsampled = subsample_dataset(sample_dataset, sample_rate)

        for split_name in sample_dataset.keys():
            original_size = len(sample_dataset[split_name])
            subsampled_size = len(subsampled[split_name])
            expected_size = int(original_size * sample_rate)

            assert (
                subsampled_size == expected_size
            ), f"Expected {expected_size} samples, got {subsampled_size}"

    def test_subsample_with_high_rate(self, sample_dataset):
        """Test subsampling with high sample rate (90%)."""
        sample_rate = 0.9
        subsampled = subsample_dataset(sample_dataset, sample_rate)

        for split_name in sample_dataset.keys():
            original_size = len(sample_dataset[split_name])
            subsampled_size = len(subsampled[split_name])
            expected_size = int(original_size * sample_rate)

            assert (
                subsampled_size == expected_size
            ), f"Expected {expected_size} samples, got {subsampled_size}"

    def test_subsample_with_rate_one(self, sample_dataset):
        """Test subsampling with sample_rate=1.0 (all data)."""
        sample_rate = 1.0
        subsampled = subsample_dataset(sample_dataset, sample_rate)

        for split_name in sample_dataset.keys():
            assert len(subsampled[split_name]) == len(
                sample_dataset[split_name]
            ), "With rate=1.0, subsampled size should equal original size"

    def test_subsample_invalid_rate_too_low(self, sample_dataset):
        """Test that sample_rate <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be between 0 and 1"):
            subsample_dataset(sample_dataset, sample_rate=0.0)

        with pytest.raises(ValueError, match="sample_rate must be between 0 and 1"):
            subsample_dataset(sample_dataset, sample_rate=-0.5)

    def test_subsample_invalid_rate_too_high(self, sample_dataset):
        """Test that sample_rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be between 0 and 1"):
            subsample_dataset(sample_dataset, sample_rate=1.5)

    def test_subsample_preserves_data_structure(self, sample_dataset):
        """Test that subsampling preserves the data structure."""
        sample_rate = 0.5
        subsampled = subsample_dataset(sample_dataset, sample_rate)

        for split_name in sample_dataset.keys():
            original_split = sample_dataset[split_name]
            subsampled_split = subsampled[split_name]

            # Check that columns are preserved
            assert set(original_split.column_names) == set(
                subsampled_split.column_names
            ), f"Split '{split_name}' columns should be preserved"

            # Check that each sample still has tokens and tags
            for sample in subsampled_split:
                assert "tokens" in sample, "Sample should have 'tokens' field"
                assert "ner_tags" in sample, "Sample should have 'ner_tags' field"
                assert len(sample["tokens"]) > 0, "Sample should have tokens"
                assert len(sample["ner_tags"]) > 0, "Sample should have tags"


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_load_and_subsample_pipeline(self):
        """Test the pipeline: load dataset -> subsample."""
        dataset = load_conll2003_dataset()
        subsampled = subsample_dataset(dataset, sample_rate=0.1)

        # Verify the pipeline executed successfully
        assert isinstance(
            subsampled, dict
        ), "Pipeline should return valid subsampled data"
        assert "train" in subsampled, "Subsampled data should contain train split"

    def test_multiple_subsampling_rates(self):
        """Test subsampling with multiple different rates."""
        dataset = load_conll2003_dataset()

        rates = [0.1, 0.25, 0.5, 0.75, 1.0]
        for rate in rates:
            subsampled = subsample_dataset(dataset, sample_rate=rate)

            assert isinstance(subsampled, dict), f"Should work with rate={rate}"
            assert len(subsampled["train"]) > 0, f"Should have data at rate={rate}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
