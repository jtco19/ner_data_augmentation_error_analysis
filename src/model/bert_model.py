"""
BERT fine-tuning module for NER (Named Entity Recognition) on CoNLL-2003 dataset.
Supports CUDA (NVIDIA GPUs), AMD GPUs via DirectML, and CPU training.
"""

import logging
import platform
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_device(force_device: Optional[str] = None) -> str:
    """
    Automatically detect and return the best available device for training.

    Supports:
    - CUDA (NVIDIA GPUs)
    - ROCm (AMD GPUs with HIP)
    - DirectML (AMD GPUs and Intel Arc on Windows)
    - CPU (fallback)

    Args:
        force_device: Force a specific device. Options: 'cuda', 'directml', 'rocm', 'cpu', None

    Returns:
        Device string to use with PyTorch

    Note:
        For DirectML (AMD GPUs on Windows):
        - Requires: pip install torch-directml
        - Works on Windows with AMD Radeon GPUs
        - May require: pip install ort-directml for ONNX Runtime optimization
    """
    if force_device:
        if force_device == "directml":
            try:
                import torch_directml

                logger.info("DirectML available and enabled")
                return torch_directml.device()
            except ImportError:
                logger.warning(
                    "DirectML requested but torch-directml not installed. "
                    "Install with: pip install torch-directml"
                )
                return _fallback_device()
        elif force_device == "rocm":
            if torch.cuda.is_available() and "rocm" in torch.version.cuda.lower():
                logger.info("ROCm GPU available")
                return "cuda"
            else:
                logger.warning("ROCm requested but not available")
                return _fallback_device()
        elif force_device == "cuda":
            if torch.cuda.is_available():
                logger.info("CUDA GPU available")
                return "cuda"
            else:
                logger.warning("CUDA requested but not available")
                return _fallback_device()
        elif force_device == "cpu":
            logger.info("CPU selected")
            return "cpu"
        else:
            logger.warning(f"Unknown device: {force_device}. Using auto-detection")

    # Auto-detection
    if torch.cuda.is_available():
        device_type = "CUDA"
        if "rocm" in torch.version.cuda.lower():
            device_type = "ROCm"
        logger.info(f"{device_type} GPU available: {torch.cuda.get_device_name(0)}")
        return "cuda"

    # Try DirectML on Windows
    if platform.system() == "Windows":
        try:
            import torch_directml

            logger.info("DirectML GPU available (AMD GPU or Intel Arc)")
            return torch_directml.device()
        except ImportError:
            logger.debug("torch-directml not available")

    # Fallback to CPU
    logger.info("Using CPU for training")
    return "cpu"


def _fallback_device() -> str:
    """Return the best available fallback device."""
    if torch.cuda.is_available():
        logger.info("Falling back to CUDA")
        return "cuda"

    if platform.system() == "Windows":
        try:
            import torch_directml

            logger.info("Falling back to DirectML")
            return torch_directml.device()
        except ImportError:
            pass

    logger.info("Falling back to CPU")
    return "cpu"


class BERTNERModel:
    """
    A class to load, fine-tune, and train BERT for Named Entity Recognition.

    Supports training on:
    - NVIDIA GPUs (CUDA)
    - AMD GPUs (ROCm or DirectML)
    - Intel Arc GPUs (DirectML on Windows)
    - CPU (fallback)
    """

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        num_labels: int = 9,
        device: Optional[str] = None,
        force_device: Optional[str] = None,
    ):
        """
        Initialize the BERT NER model.

        Args:
            model_name: The name of the BERT model from HuggingFace Hub
            num_labels: Number of NER labels (for CoNLL-2003 this is typically 9)
            device: Explicit device to use. If None, auto-detects.
                   Options: 'cuda', 'cpu', 'directml', or DirectML device object
            force_device: Force a specific device type. Options: 'cuda', 'directml', 'rocm', 'cpu'
                         Takes precedence over device parameter.

        Examples:
            # Auto-detect device (CUDA > DirectML > CPU)
            model = BERTNERModel()

            # Force AMD GPU with DirectML
            model = BERTNERModel(force_device='directml')

            # Force CUDA
            model = BERTNERModel(force_device='cuda')

            # Use CPU
            model = BERTNERModel(force_device='cpu')
        """
        # Determine device to use
        if force_device:
            self.device = get_device(force_device=force_device)
        elif device:
            self.device = device
        else:
            self.device = get_device()

        self.model_name = model_name
        self.num_labels = num_labels
        self.use_directml = "directml" in str(self.device).lower()

        logger.info(f"Initializing BERT model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of labels: {num_labels}")
        if self.use_directml:
            logger.info("DirectML acceleration enabled for AMD GPU/Intel Arc")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer loaded from {model_name}")

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(self.device)
        logger.info(f"Model loaded and moved to {self.device}")

    def tokenize_and_align_labels(
        self,
        examples: Dict,
        label_all_tokens: bool = True,
    ) -> Dict:
        """
        Tokenize text and align labels with tokens.

        This function handles the case where a single word may be split into
        multiple tokens by the tokenizer. Labels are aligned accordingly.

        Args:
            examples: Dictionary containing 'tokens', 'ner_tags' from the dataset
            label_all_tokens: If True, all subword tokens get the same label.
                             If False, only the first token of a word gets the label.

        Returns:
            Dictionary with tokenized input and aligned labels
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=512,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens get -100 (ignored in loss computation)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First token of a word gets the label
                    label_ids.append(label[word_idx])
                else:
                    # Subsequent tokens of same word
                    if label_all_tokens:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_dataset(
        self,
        dataset_split: Dataset,
        batch_size: int = 32,
        label_all_tokens: bool = True,
    ) -> Dataset:
        """
        Prepare dataset for training by tokenizing and aligning labels.

        Args:
            dataset_split: A Dataset object (train, validation, or test split)
            batch_size: Batch size for tokenization
            label_all_tokens: Whether to label all tokens or just first token of word

        Returns:
            A tokenized Dataset ready for training
        """
        logger.info(f"Preparing dataset with {len(dataset_split)} samples")

        # Tokenize and align labels
        tokenized_dataset = dataset_split.map(
            lambda examples: self.tokenize_and_align_labels(examples, label_all_tokens),
            batched=True,
            batch_size=batch_size,
            desc="Tokenizing and aligning labels",
        )

        logger.info(
            f"Dataset preparation complete. Total samples: {len(tokenized_dataset)}"
        )
        return tokenized_dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./results",
        num_epochs: int = 3,
        per_device_batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        save_strategy: str = "epoch",
        logging_steps: int = 100,
        eval_strategy: str = "epoch",
    ) -> Tuple[object, Dict]:
        """
        Fine-tune BERT model on NER task.

        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Optional tokenized evaluation dataset
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            per_device_batch_size: Batch size per GPU/CPU
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps for learning rate scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            save_strategy: Strategy for saving checkpoints ('epoch', 'steps', 'no')
            logging_steps: How often to log training metrics
            eval_strategy: Strategy for evaluation ('epoch', 'steps', 'no')

        Returns:
            A tuple of (trainer object, training results dictionary)
        """
        logger.info("Starting training...")
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        # Data collator for token classification
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Determine mixed precision setting based on device
        use_mixed_precision = False
        if torch.cuda.is_available():
            # CUDA supports FP16
            use_mixed_precision = True
            logger.info("Mixed precision (FP16) enabled for CUDA")
        elif self.use_directml:
            # DirectML has limited FP16 support; use FP32 for stability
            use_mixed_precision = False
            logger.info("Mixed precision disabled for DirectML (using FP32)")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_strategy="steps",
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy if eval_dataset else "no",
            save_total_limit=3,
            seed=42,
            fp16=use_mixed_precision,
            report_to=[],  # Disable wandb/other integrations
            dataloader_pin_memory=not self.use_directml,  # DirectML prefers pin_memory=False
            remove_unused_columns=True,  # Remove columns not needed by model (tokens, ner_tags, etc)
        )

        logger.info(f"Training arguments:\n{training_args}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Mixed precision: {use_mixed_precision}")

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        logger.info("Trainer initialized. Starting training...")

        # Train
        train_result = trainer.train()

        logger.info(f"Training complete!")
        logger.info(f"Training results: {train_result.metrics}")

        return trainer, train_result.metrics

    def compute_metrics(self, p):
        """
        Compute F1, precision, and recall for token classification.

        Args:
            p: EvalPrediction object containing predictions and label_ids

        Returns:
            Dictionary with F1, precision, and recall scores
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        predictions = p.predictions
        labels = p.label_ids

        # Get the predicted class for each token
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens with label -100)
        true_predictions = [
            [p for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        # Flatten lists
        true_predictions_flat = []
        true_labels_flat = []
        for pred_seq, label_seq in zip(true_predictions, true_labels):
            true_predictions_flat.extend(pred_seq)
            true_labels_flat.extend(label_seq)

        # Calculate metrics
        f1 = f1_score(
            true_labels_flat,
            true_predictions_flat,
            average="weighted",
            zero_division=0,
        )
        precision = precision_score(
            true_labels_flat,
            true_predictions_flat,
            average="weighted",
            zero_division=0,
        )
        recall = recall_score(
            true_labels_flat,
            true_predictions_flat,
            average="weighted",
            zero_division=0,
        )

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def evaluate(
        self,
        eval_dataset: Dataset,
    ) -> Dict:
        """
        Evaluate the model on an evaluation dataset.

        Args:
            eval_dataset: Tokenized evaluation dataset

        Returns:
            Dictionary containing evaluation metrics (loss, F1, precision, recall)
        """
        logger.info(f"Starting evaluation on {len(eval_dataset)} samples...")

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        eval_args = TrainingArguments(
            output_dir="./eval_results",
            per_device_eval_batch_size=8,
        )

        trainer = Trainer(
            model=self.model,
            args=eval_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        eval_result = trainer.evaluate(eval_dataset)

        logger.info(f"Evaluation results: {eval_result}")
        return eval_result

    def save_model(self, save_path: str) -> None:
        """
        Save the fine-tuned model and tokenizer.

        Args:
            save_path: Path to save the model and tokenizer
        """
        logger.info(f"Saving model and tokenizer to {save_path}")

        # Move model to CPU for saving (required for DirectML compatibility)
        original_device = self.device
        self.model.to("cpu")

        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info("Model and tokenizer saved successfully")
        finally:
            # Move model back to original device
            self.model.to(original_device)
            logger.info(f"Model moved back to {original_device}")

    def load_model(self, model_path: str) -> None:
        """
        Load a previously saved model and tokenizer.

        Args:
            model_path: Path to the saved model
        """
        logger.info(f"Loading model and tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        logger.info("Model and tokenizer loaded successfully")


def create_bert_ner_model(
    model_name: str = "bert-base-cased",
    num_labels: int = 9,
    force_device: Optional[str] = None,
) -> BERTNERModel:
    """
    Factory function to create a BERT NER model.

    Args:
        model_name: Name of the BERT model from HuggingFace Hub
        num_labels: Number of NER labels
        force_device: Force a specific device ('cuda', 'directml', 'rocm', 'cpu')

    Returns:
        A BERTNERModel instance

    Examples:
        # Auto-detect device
        model = create_bert_ner_model()

        # Use DirectML for AMD GPU
        model = create_bert_ner_model(force_device='directml')
    """
    logger.info(f"Creating BERT NER model: {model_name} with {num_labels} labels")
    return BERTNERModel(
        model_name=model_name, num_labels=num_labels, force_device=force_device
    )


def check_gpu_availability() -> Dict[str, bool]:
    """
    Check availability of different GPU acceleration backends.

    Returns:
        Dictionary with availability status for each backend
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "directml_available": False,
        "rocm_available": False,
        "windows": platform.system() == "Windows",
    }

    # Check CUDA
    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["rocm_available"] = "rocm" in torch.version.cuda.lower()

    # Check DirectML
    try:
        import torch_directml

        info["directml_available"] = True
    except ImportError:
        pass

    return info


def print_gpu_setup_instructions() -> None:
    """
    Print setup instructions for different GPU types.
    """
    gpu_info = check_gpu_availability()

    print("\n" + "=" * 80)
    print("GPU SETUP INFORMATION")
    print("=" * 80)

    print("\nAvailable Backends:")
    print(f"  CUDA (NVIDIA): {gpu_info['cuda_available']}")
    if gpu_info["cuda_available"]:
        print(f"    Device: {gpu_info.get('cuda_device_name', 'Unknown')}")

    print(f"  ROCm (AMD - Linux): {gpu_info['rocm_available']}")
    print(f"  DirectML (Windows): {gpu_info['directml_available']}")
    print(f"  Windows OS: {gpu_info['windows']}")

    print("\nSetup Instructions:")

    if not gpu_info["cuda_available"]:
        print("\n  CUDA (NVIDIA GPU):")
        print("    1. Install NVIDIA CUDA Toolkit")
        print("    2. Install cuDNN")
        print(
            "    3. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )

    if not gpu_info["rocm_available"] and not gpu_info["cuda_available"]:
        print("\n  ROCm (AMD GPU on Linux):")
        print("    1. Install ROCm from https://rocmdocs.amd.com/")
        print(
            "    2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7"
        )

    if gpu_info["windows"] and not gpu_info["directml_available"]:
        print("\n  DirectML (AMD GPU / Intel Arc on Windows):")
        print("    1. pip install torch-directml")
        print(
            "    2. Optional: pip install ort-directml (for ONNX Runtime optimization)"
        )

    print("\nUsage with Different Devices:")
    print("  # Auto-detect:")
    print("  model = create_bert_ner_model()")
    print("")
    print("  # Force CUDA:")
    print("  model = create_bert_ner_model(force_device='cuda')")
    print("")
    print("  # Force DirectML (AMD GPU on Windows):")
    print("  model = create_bert_ner_model(force_device='directml')")
    print("")
    print("  # Force CPU:")
    print("  model = create_bert_ner_model(force_device='cpu')")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    logger.info("BERT NER Model Module")
    logger.info("Import this module to use BERTNERModel or create_bert_ner_model()")
    # print_gpu_setup_instructions()
    device = _fallback_device()
    logger.info(f"Best available device for training: {device}")
