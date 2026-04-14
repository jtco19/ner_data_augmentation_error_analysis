"""
Comprehensive test suite for bert_model.py using pytest.
"""

import sys
import os
from pathlib import Path
import pytest
import tempfile
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from unittest.mock import Mock, patch, MagicMock
import logging

# Add the model directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from bert_model import (
    get_device,
    _fallback_device,
    BERTNERModel,
    create_bert_ner_model,
    check_gpu_availability,
)

DEVICE_OPTIONS = [
    "ipu",
    "xpu",
    "mkldnn",
    "opengl",
    "opencl",
    "ideep",
    "hip",
    "ve",
    "fpga",
    "maia",
    "xla",
    "lazy",
    "vulkan",
    "mps",
    "meta",
    "hpu",
    "mtia",
    "privateuseone",
    "privateusetwo",
    "privateusethree",
    "directml",
    "rocm",
    "cuda",
    "cpu",
]

# Get logger for tests (pytest will handle configuration)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_ner_dataset():
    """
    Create a mock NER dataset for testing.

    Returns:
        A DatasetDict with train, validation, and test splits.
    """
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

        for _ in range(num_samples):
            # Random number of tokens between 5 and 15
            num_tokens = np.random.randint(5, 16)
            tokens = [f"word_{i}" for i in range(num_tokens)]
            tags = [np.random.randint(0, len(label_names)) for _ in range(num_tokens)]

            tokens_list.append(tokens)
            tags_list.append(tags)

        data = {
            "tokens": tokens_list,
            "ner_tags": tags_list,
        }

        return Dataset.from_dict(data)

    train_split = create_split(20, seed=42)
    validation_split = create_split(5, seed=43)
    test_split = create_split(5, seed=44)

    dataset_dict = DatasetDict(
        {
            "train": train_split,
            "validation": validation_split,
            "test": test_split,
        }
    )

    return dataset_dict


class TestDeviceDetection:
    """Tests for device detection functions."""

    def test_get_device_returns_string(self):
        """Test that get_device returns a string or device object."""
        device = get_device()
        assert device is not None, "Device should not be None"

    def test_get_device_force_cpu(self):
        """Test forcing CPU device."""
        device = get_device(force_device="cpu")
        assert device == "cpu", "Should force CPU device"

    def test_get_device_force_cuda(self):
        """Test forcing CUDA device."""
        if torch.cuda.is_available():
            device = get_device(force_device="cuda")
            assert device == "cuda", "Should force CUDA when available"
        else:
            # If CUDA not available, should fall back gracefully
            device = get_device(force_device="cuda")
            assert device is not None, "Should return a fallback device"

    def test_get_device_invalid_device_falls_back(self):
        """Test that invalid device forces falls back gracefully."""
        device = get_device(force_device="invalid_device")
        assert device is not None, "Should return fallback device for invalid input"

    def test_fallback_device_returns_valid_device(self):
        """Test that fallback device returns a valid device."""
        device = _fallback_device()
        assert (
            str(device).split(":")[0].lower() in DEVICE_OPTIONS
        ), "Should return valid device"


class TestGPUAvailability:
    """Tests for GPU availability checking."""

    def test_check_gpu_availability_returns_dict(self):
        """Test that GPU availability check returns a dictionary."""
        info = check_gpu_availability()
        assert isinstance(info, dict), "Should return a dictionary"

    def test_check_gpu_availability_has_required_keys(self):
        """Test that GPU info has required keys."""
        info = check_gpu_availability()
        required_keys = {
            "cuda_available",
            "directml_available",
            "rocm_available",
            "windows",
        }
        assert required_keys.issubset(info.keys()), f"Should have keys: {required_keys}"

    def test_check_gpu_availability_values_are_booleans(self):
        """Test that GPU availability values are booleans."""
        info = check_gpu_availability()
        assert isinstance(info["cuda_available"], bool), "cuda_available should be bool"
        assert isinstance(
            info["directml_available"], bool
        ), "directml_available should be bool"
        assert isinstance(info["windows"], bool), "windows should be bool"


class TestBERTNERModelInitialization:
    """Tests for BERTNERModel initialization."""

    def test_model_initialization_with_defaults(self):
        """Test BERTNERModel initialization with default parameters."""
        model = BERTNERModel()
        assert model.model_name == "bert-base-cased", "Should use default model name"
        assert model.num_labels == 9, "Should use default num_labels"
        assert model.tokenizer is not None, "Tokenizer should be initialized"
        assert model.model is not None, "Model should be initialized"

    def test_model_initialization_with_custom_labels(self):
        """Test BERTNERModel initialization with custom number of labels."""
        num_labels = 5
        model = BERTNERModel(num_labels=num_labels)
        assert (
            model.num_labels == num_labels
        ), f"Should use custom num_labels={num_labels}"

    def test_model_initialization_device_is_set(self):
        """Test that device is properly set during initialization."""
        model = BERTNERModel()
        assert model.device is not None, "Device should be set"

    def test_model_initialization_force_cpu(self):
        """Test BERTNERModel initialization forcing CPU."""
        model = BERTNERModel(force_device="cpu")
        assert model.device == "cpu", "Should force CPU device"

    def test_model_initialization_use_directml_flag(self):
        """Test that use_directml flag is set correctly."""
        model = BERTNERModel()
        assert isinstance(model.use_directml, bool), "use_directml should be boolean"


class TestTokenizationAndAlignment:
    """Tests for tokenization and label alignment."""

    def test_tokenize_and_align_labels_returns_dict(self, sample_ner_dataset):
        """Test that tokenization returns a dictionary-like object."""
        model = BERTNERModel()
        train_split = sample_ner_dataset["train"]

        # Use first example
        example = {
            "tokens": [train_split["tokens"][0]],
            "ner_tags": [train_split["ner_tags"][0]],
        }

        result = model.tokenize_and_align_labels(example)
        # Check for dict-like behavior (BatchFeature is dict-like)
        assert hasattr(result, "__getitem__"), "Should return dict-like object"
        assert hasattr(result, "keys"), "Should have keys method"

    def test_tokenize_and_align_labels_has_required_keys(self, sample_ner_dataset):
        """Test that tokenized output has required keys."""
        model = BERTNERModel()
        train_split = sample_ner_dataset["train"]

        example = {
            "tokens": [train_split["tokens"][0]],
            "ner_tags": [train_split["ner_tags"][0]],
        }

        result = model.tokenize_and_align_labels(example)
        required_keys = {"input_ids", "attention_mask", "labels"}
        assert required_keys.issubset(
            result.keys()
        ), f"Should have keys: {required_keys}"

    def test_tokenize_and_align_labels_preserves_structure(self, sample_ner_dataset):
        """Test that tokenization preserves input/output structure."""
        model = BERTNERModel()
        train_split = sample_ner_dataset["train"]

        example = {
            "tokens": [train_split["tokens"][0]],
            "ner_tags": [train_split["ner_tags"][0]],
        }

        result = model.tokenize_and_align_labels(example)

        assert len(result["input_ids"]) == len(
            result["labels"]
        ), "Input IDs and labels should have same length"


class TestDatasetPreparation:
    """Tests for dataset preparation."""

    def test_prepare_dataset_returns_dataset(self, sample_ner_dataset):
        """Test that prepare_dataset returns a Dataset object."""
        model = BERTNERModel()
        train_split = sample_ner_dataset["train"]

        prepared = model.prepare_dataset(train_split, batch_size=2)
        assert isinstance(prepared, Dataset), "Should return a Dataset"

    def test_prepare_dataset_same_size(self, sample_ner_dataset):
        """Test that prepared dataset has same number of samples."""
        model = BERTNERModel()
        train_split = sample_ner_dataset["train"]
        original_size = len(train_split)

        prepared = model.prepare_dataset(train_split, batch_size=2)
        assert (
            len(prepared) == original_size
        ), f"Prepared dataset should have {original_size} samples"

    def test_prepare_dataset_has_labels(self, sample_ner_dataset):
        """Test that prepared dataset has labels."""
        model = BERTNERModel()
        train_split = sample_ner_dataset["train"]

        prepared = model.prepare_dataset(train_split, batch_size=2)

        # Check first sample
        sample = prepared[0]
        assert "labels" in sample, "Sample should have labels"


class TestModelSaveLoad:
    """Tests for model saving and loading."""

    def test_save_model(self):
        """Test that model can be saved."""
        model = BERTNERModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            model.save_model(save_path)

            assert os.path.exists(save_path), "Model directory should be created"
            # Check that model files were created (pytorch_model.bin or model.safetensors)
            files = os.listdir(save_path)
            assert len(files) > 0, "Save directory should contain files"

            # Check for key files that should exist
            has_model_file = any(
                f in files for f in ["pytorch_model.bin", "model.safetensors"]
            )
            has_config = "config.json" in files
            has_tokenizer_config = "tokenizer_config.json" in files

            assert (
                has_model_file or has_config
            ), "Should have model weights and/or config file"
            logger.info(f"Saved files: {files}")

    def test_load_model(self):
        """Test that model can be loaded from saved path."""
        model1 = BERTNERModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            model1.save_model(save_path)

            # Create new model and load
            model2 = BERTNERModel()
            # Should not raise any errors
            model2.load_model(save_path)

            assert model2.model is not None, "Loaded model should not be None"
            assert model2.tokenizer is not None, "Loaded tokenizer should not be None"

    def test_save_load_roundtrip(self):
        """Test complete save/load roundtrip with inference."""
        model1 = BERTNERModel(num_labels=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            model1.save_model(save_path)

            # Create new model and load
            model2 = BERTNERModel()
            model2.load_model(save_path)

            # Move to CPU for inference (DirectML tensors can't be used directly)
            model2.model.to("cpu")

            # Verify loaded model can do inference
            input_ids = torch.randint(0, 1000, (2, 10))
            attention_mask = torch.ones((2, 10))

            with torch.no_grad():
                outputs = model2.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            assert outputs is not None, "Loaded model should produce outputs"
            assert (
                outputs.logits.shape[2] == 10
            ), "Output should have correct number of labels"


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_bert_ner_model_returns_instance(self):
        """Test that factory function returns BERTNERModel instance."""
        model = create_bert_ner_model()
        assert isinstance(model, BERTNERModel), "Should return BERTNERModel instance"

    def test_create_bert_ner_model_with_custom_labels(self):
        """Test factory function with custom parameters."""
        num_labels = 15
        model = create_bert_ner_model(num_labels=num_labels)
        assert model.num_labels == num_labels, f"Should have {num_labels} labels"

    def test_create_bert_ner_model_force_device(self):
        """Test factory function with forced device."""
        model = create_bert_ner_model(force_device="cpu")
        assert model.device == "cpu", "Should force CPU device"


class TestIntegration:
    """Integration tests."""

    def test_model_initialization_and_dataset_preparation(self, sample_ner_dataset):
        """Test complete pipeline: initialization -> data preparation."""
        model = BERTNERModel()
        train_split = sample_ner_dataset["train"]

        prepared = model.prepare_dataset(train_split, batch_size=2)

        assert prepared is not None, "Should prepare dataset successfully"
        assert len(prepared) > 0, "Prepared dataset should have samples"

    def test_multiple_model_instances(self, sample_ner_dataset):
        """Test creating multiple model instances."""
        model1 = create_bert_ner_model(num_labels=9)
        model2 = create_bert_ner_model(num_labels=5)

        assert model1.num_labels == 9, "Model 1 should have 9 labels"
        assert model2.num_labels == 5, "Model 2 should have 5 labels"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_model_with_small_max_length(self):
        """Test tokenization with small max_length."""
        model = BERTNERModel()

        # Create very long sequence
        tokens = {"tokens": [["word"] * 100], "ner_tags": [[0] * 100]}
        result = model.tokenize_and_align_labels(tokens)

        # Should truncate to max length
        assert len(result["input_ids"][0]) == 512, "Should use max length of 512"

    def test_model_forward_pass_does_not_error(self):
        """Test that model can do a forward pass without error."""
        model = BERTNERModel()

        # Move to CPU for inference (DirectML tensors can't be used directly)
        model.model.to("cpu")

        # Create dummy input
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones((2, 10))

        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs is not None, "Forward pass should return outputs"
        assert hasattr(outputs, "logits"), "Outputs should have logits"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
