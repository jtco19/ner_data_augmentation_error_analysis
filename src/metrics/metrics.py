"""
Evaluation metrics for assessing the quality of augmented NER datasets and model performance.
Focuses on measuring data augmentation corruption, label consistency, and per-entity-type F1 scores.
"""

import logging
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

# Configure logging
logger = logging.getLogger(__name__)


def extract_entities_from_labels(
    tokens: List[str], labels: List[str]
) -> List[Tuple[int, int, str]]:
    """
    Extract entity spans from BIO-tagged token lists.

    Args:
        tokens: List of tokens
        labels: List of BIO tags (e.g., ['B-PER', 'I-PER', 'O', 'B-LOC'])

    Returns:
        List of tuples (start_idx, end_idx, entity_type) representing entity spans
    """
    entities = []
    current_entity = None
    start_idx = None

    for i, label in enumerate(labels):
        if label == "O":
            if current_entity:
                entities.append((start_idx, i - 1, current_entity))
                current_entity = None
                start_idx = None
        else:
            tag_prefix, entity_type = label.split("-") if "-" in label else ("O", label)

            if tag_prefix == "B":
                if current_entity:
                    entities.append((start_idx, i - 1, current_entity))
                current_entity = entity_type
                start_idx = i
            elif tag_prefix == "I":
                if current_entity != entity_type:
                    if current_entity:
                        entities.append((start_idx, i - 1, current_entity))
                    current_entity = entity_type
                    start_idx = i

    if current_entity:
        entities.append((start_idx, len(labels) - 1, current_entity))

    return entities


def span_corruption_rate(
    original_sentences: List[List[str]],
    original_labels: List[List[str]],
    augmented_sentences: List[List[str]],
    augmented_labels: List[List[str]],
    alignment_mapping: List[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute the Span-Corruption Rate: percentage of augmented sentences where
    an original entity span's label is altered or lost.

    A span is considered corrupted if:
    - The span is completely lost (tokens don't exist in augmented version)
    - The span's label is changed
    - Part of the span is lost or has a different label

    Args:
        original_sentences: List of original token sequences
        original_labels: List of original BIO label sequences
        augmented_sentences: List of augmented token sequences
        augmented_labels: List of augmented BIO label sequences
        alignment_mapping: Optional list mapping augmented token indices to original indices
                          If None, uses simple token matching

    Returns:
        Dict with keys:
        - 'span_corruption_rate': float (0-1) percentage of augmented sentences with corruptions
        - 'total_augmented_sentences': int
        - 'corrupted_sentences': int
        - 'total_spans': int
        - 'corrupted_spans': int
    """
    logger.info(
        f"Starting span corruption analysis on {len(augmented_sentences)} sentences"
    )
    if len(original_sentences) != len(augmented_sentences):
        logger.error("Original and augmented sentences have different lengths")
        raise ValueError("Original and augmented sentences must have same length")

    corrupted_sentences = 0
    corrupted_spans = 0
    total_spans = 0

    for idx, (orig_sent, orig_labels, aug_sent, aug_labels, alignment) in enumerate(
        zip(
            original_sentences,
            original_labels,
            augmented_sentences,
            augmented_labels,
            (
                alignment_mapping
                if alignment_mapping
                else [None] * len(original_sentences)
            ),
        )
    ):
        # Extract entity spans from original and augmented versions
        orig_entities = extract_entities_from_labels(orig_sent, orig_labels)
        aug_entities = extract_entities_from_labels(aug_sent, aug_labels)

        # Convert augmented entities to set for fast lookup
        aug_entity_set = set(aug_entities)

        sentence_has_corruption = False

        for start, end, entity_type in orig_entities:
            total_spans += 1

            # Get the original span text and tokens
            orig_span_tokens = orig_sent[start : end + 1]
            orig_span_text = " ".join(orig_span_tokens)

            # Check if this span exists in augmented version with same label
            span_found = False
            for aug_start, aug_end, aug_type in aug_entities:
                aug_span_tokens = aug_sent[aug_start : aug_end + 1]
                aug_span_text = " ".join(aug_span_tokens)

                if orig_span_text == aug_span_text and entity_type == aug_type:
                    span_found = True
                    break

            if not span_found:
                corrupted_spans += 1
                sentence_has_corruption = True
                logger.debug(
                    f"Sentence {idx}: Found corrupted span '{orig_span_text}' ({entity_type})"
                )

        if sentence_has_corruption:
            corrupted_sentences += 1

    total_augmented = len(augmented_sentences)
    corruption_rate = (
        corrupted_sentences / total_augmented if total_augmented > 0 else 0.0
    )

    logger.info(
        f"Span corruption analysis complete: {corrupted_sentences}/{total_augmented} sentences corrupted "
        f"({corruption_rate*100:.2f}%), {corrupted_spans}/{total_spans} spans corrupted"
    )

    return {
        "span_corruption_rate": corruption_rate,
        "total_augmented_sentences": total_augmented,
        "corrupted_sentences": corrupted_sentences,
        "total_spans": total_spans,
        "corrupted_spans": corrupted_spans,
    }


def label_flip_rate(
    original_labels: List[List[str]],
    augmented_labels: List[List[str]],
    alignment_mapping: List[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute the Label-Flip Rate: percentage of tokens whose label switches from
    entity → non-entity (or vice versa) incorrectly.

    A flip is counted when a token's label changes between:
    - Entity label (B-* or I-*) → Non-entity (O)
    - Non-entity (O) → Entity label (B-* or I-*)

    Args:
        original_labels: List of original BIO label sequences
        augmented_labels: List of augmented BIO label sequences
        alignment_mapping: Optional list mapping augmented token indices to original indices
                          If None, assumes 1-to-1 correspondence where sentences have same length

    Returns:
        Dict with keys:
        - 'label_flip_rate': float (0-1) percentage of flipped tokens
        - 'total_tokens': int
        - 'flipped_tokens': int
        - 'entity_to_non_entity': int (count of O labels lost)
        - 'non_entity_to_entity': int (count of O labels gained)
    """
    logger.info(f"Starting label flip analysis on {len(augmented_labels)} sequences")
    if len(original_labels) != len(augmented_labels):
        logger.error("Original and augmented labels have different lengths")
        raise ValueError("Original and augmented labels must have same length")

    total_tokens = 0
    flipped_tokens = 0
    entity_to_non_entity = 0
    non_entity_to_entity = 0

    for orig_seq, aug_seq in zip(original_labels, augmented_labels):
        min_len = min(len(orig_seq), len(aug_seq))

        for i in range(min_len):
            orig_label = orig_seq[i]
            aug_label = aug_seq[i]

            is_orig_entity = orig_label != "O"
            is_aug_entity = aug_label != "O"

            if is_orig_entity != is_aug_entity:
                total_tokens += 1
                flipped_tokens += 1

                if is_orig_entity and not is_aug_entity:
                    entity_to_non_entity += 1
                else:
                    non_entity_to_entity += 1
            else:
                total_tokens += 1

    flip_rate = flipped_tokens / total_tokens if total_tokens > 0 else 0.0

    logger.info(
        f"Label flip analysis complete: {flipped_tokens}/{total_tokens} tokens flipped ({flip_rate*100:.2f}%), "
        f"Entity→Non-Entity: {entity_to_non_entity}, Non-Entity→Entity: {non_entity_to_entity}"
    )

    return {
        "label_flip_rate": flip_rate,
        "total_tokens": total_tokens,
        "flipped_tokens": flipped_tokens,
        "entity_to_non_entity": entity_to_non_entity,
        "non_entity_to_entity": non_entity_to_entity,
    }


def per_type_f1_scores(
    y_true: List[List[str]], y_pred: List[List[str]], entity_types: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute F1, precision, and recall scores for each entity type separately.

    Uses the seqeval library to properly handle BIO-tagged sequences and compute
    entity-level metrics (not token-level).

    Args:
        y_true: List of true label sequences (BIO format)
        y_pred: List of predicted label sequences (BIO format)
        entity_types: Optional list of entity types to include. If None, extracts from labels.
                     Common types: ['PER', 'LOC', 'ORG', 'MISC']

    Returns:
        Dict mapping entity_type to dict with keys:
        - 'f1': float F1 score
        - 'precision': float precision
        - 'recall': float recall
        - 'support': int number of true entities of this type
    """
    logger.info(f"Computing per-type F1 scores for {len(y_true)} sequences")
    # Extract entity types from labels if not provided
    if entity_types is None:
        entity_types_set = set()
        for seq in y_true + y_pred:
            for label in seq:
                if label != "O" and "-" in label:
                    entity_types_set.add(label.split("-")[1])
        entity_types = sorted(list(entity_types_set))

    logger.debug(f"Entity types found: {entity_types}")

    results = {}

    # Overall metrics
    overall_precision = precision_score(y_true, y_pred, zero_division=0)
    overall_recall = recall_score(y_true, y_pred, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(
        f"Overall F1: {overall_f1:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}"
    )

    results["overall"] = {
        "f1": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
    }

    # Per-type metrics
    for entity_type in entity_types:
        logger.debug(f"Computing metrics for entity type: {entity_type}")
        # Extract only this entity type from predictions and ground truth
        y_true_type = []
        y_pred_type = []

        for true_seq, pred_seq in zip(y_true, y_pred):
            true_seq_filtered = []
            pred_seq_filtered = []

            for true_label, pred_label in zip(true_seq, pred_seq):
                # Keep entity if it matches this type, otherwise mark as 'O'
                if "-" in true_label and true_label.split("-")[1] == entity_type:
                    true_seq_filtered.append(true_label)
                else:
                    true_seq_filtered.append("O")

                if "-" in pred_label and pred_label.split("-")[1] == entity_type:
                    pred_seq_filtered.append(pred_label)
                else:
                    pred_seq_filtered.append("O")

            y_true_type.append(true_seq_filtered)
            y_pred_type.append(pred_seq_filtered)

        try:
            precision = precision_score(y_true_type, y_pred_type, zero_division=0)
            recall = recall_score(y_true_type, y_pred_type, zero_division=0)
            f1 = f1_score(y_true_type, y_pred_type, zero_division=0)
        except:
            # If there are no entities of this type, scores are undefined
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            logger.warning(
                f"Could not compute metrics for entity type {entity_type} (possibly no entities)"
            )

        # Count support (number of true entities of this type)
        support = sum(
            sum(
                1
                for label in seq
                if "-" in label and label.split("-")[1] == entity_type
            )
            for seq in y_true
        )

        logger.info(
            f"  {entity_type}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={support}"
        )

        results[entity_type] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": support,
        }

    return results


def augmentation_quality_report(
    original_sentences: List[List[str]],
    original_labels: List[List[str]],
    augmented_sentences: List[List[str]],
    augmented_labels: List[List[str]],
    predictions: List[List[str]] = None,
    entity_types: List[str] = None,
) -> Dict:
    """
    Generate a comprehensive augmentation quality report combining all metrics.

    Args:
        original_sentences: List of original token sequences
        original_labels: List of original BIO label sequences
        augmented_sentences: List of augmented token sequences
        augmented_labels: List of augmented BIO label sequences
        predictions: Optional predicted labels for per-type F1 computation
        entity_types: Optional list of entity types to track

    Returns:
        Dict containing:
        - 'span_corruption': dict from span_corruption_rate()
        - 'label_flip': dict from label_flip_rate()
        - 'per_type_f1': dict from per_type_f1_scores() (if predictions provided)
    """
    logger.info("=" * 60)
    logger.info("Generating comprehensive augmentation quality report")
    logger.info(f"Processing {len(augmented_sentences)} augmented samples")
    logger.info("=" * 60)

    report = {
        "span_corruption": span_corruption_rate(
            original_sentences, original_labels, augmented_sentences, augmented_labels
        ),
        "label_flip": label_flip_rate(original_labels, augmented_labels),
    }

    if predictions is not None:
        logger.info("Computing per-type F1 scores...")
        report["per_type_f1"] = per_type_f1_scores(
            augmented_labels, predictions, entity_types
        )

    logger.info("Augmentation quality report complete")
    logger.info("=" * 60)

    return report
