"""
Evaluation metrics for augmented NER datasets and model performance.

The data-quality metrics are intentionally entity-aware: NER augmentation can
look fluent while still breaking BIO spans, deleting entity tokens, or assigning
labels to the wrong aligned token after insertion/deletion.
"""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from seqeval.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


EntitySpan = Tuple[int, int, str]


def normalize_entity_type(label: Optional[str]) -> Optional[str]:
    """
    Normalize a BIO label to its entity type.

    Returns None for non-entity labels. Examples:
    - B-PER -> PER
    - I-ORG -> ORG
    - O -> None
    """
    if label is None:
        return None

    label = str(label)
    if label == "O" or label == "":
        return None
    if "-" in label:
        prefix, entity_type = label.split("-", 1)
        if prefix in {"B", "I"} and entity_type:
            return entity_type
    return label


def extract_entities_from_labels(
    tokens: List[str], labels: List[str]
) -> List[EntitySpan]:
    """
    Extract entity spans from BIO-tagged token lists.

    Invalid I-* starts are treated as B-* starts so that noisy augmented labels
    can still be measured instead of crashing the metric pipeline.
    """
    entities: List[EntitySpan] = []
    current_entity: Optional[str] = None
    start_idx: Optional[int] = None

    if len(tokens) != len(labels):
        logger.debug(
            "Token/label length mismatch while extracting entities: %s tokens, %s labels",
            len(tokens),
            len(labels),
        )

    for i, label in enumerate(labels[: len(tokens)]):
        entity_type = normalize_entity_type(label)
        prefix = str(label).split("-", 1)[0] if "-" in str(label) else None

        if entity_type is None:
            if current_entity is not None:
                entities.append((start_idx, i - 1, current_entity))
                current_entity = None
                start_idx = None
            continue

        starts_new_entity = (
            prefix == "B"
            or current_entity is None
            or current_entity != entity_type
        )

        if starts_new_entity:
            if current_entity is not None:
                entities.append((start_idx, i - 1, current_entity))
            current_entity = entity_type
            start_idx = i

    if current_entity is not None:
        entities.append((start_idx, min(len(tokens), len(labels)) - 1, current_entity))

    return entities


# Short alias matching the project proposal language.
extract_entities = extract_entities_from_labels


def align_original_and_augmented_tokens(
    original_tokens: List[str], augmented_tokens: List[str]
) -> Dict[str, object]:
    """
    Align original and augmented tokens with an edit-script heuristic.

    Returns both directions:
    - orig_to_aug[i] is the aligned augmented index for original token i, or None
      if that original token was deleted.
    - aug_to_orig[j] is the aligned original index for augmented token j, or None
      if that augmented token was inserted.

    Exact equal blocks are aligned directly. Replace blocks with equal or partly
    equal length are aligned positionally so substitutions count as changed
    tokens rather than as unrelated delete+insert operations.
    """
    matcher = SequenceMatcher(
        a=original_tokens,
        b=augmented_tokens,
        autojunk=False,
    )
    orig_to_aug: List[Optional[int]] = [None] * len(original_tokens)
    aug_to_orig: List[Optional[int]] = [None] * len(augmented_tokens)
    opcodes = matcher.get_opcodes()

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "insert":
            continue
        if tag == "delete":
            continue

        span_len = min(i2 - i1, j2 - j1)
        for offset in range(span_len):
            orig_idx = i1 + offset
            aug_idx = j1 + offset
            orig_to_aug[orig_idx] = aug_idx
            aug_to_orig[aug_idx] = orig_idx

    return {
        "orig_to_aug": orig_to_aug,
        "aug_to_orig": aug_to_orig,
        "opcodes": opcodes,
    }


def _alignment_from_aug_to_orig(
    original_len: int,
    augmented_len: int,
    aug_to_orig: Optional[List[int]],
) -> Dict[str, object]:
    orig_to_aug: List[Optional[int]] = [None] * original_len
    normalized_aug_to_orig: List[Optional[int]] = [None] * augmented_len

    if aug_to_orig is None:
        for idx in range(min(original_len, augmented_len)):
            orig_to_aug[idx] = idx
            normalized_aug_to_orig[idx] = idx
        return {
            "orig_to_aug": orig_to_aug,
            "aug_to_orig": normalized_aug_to_orig,
            "opcodes": [],
        }

    for aug_idx, orig_idx in enumerate(aug_to_orig[:augmented_len]):
        if orig_idx is None:
            continue
        if 0 <= orig_idx < original_len:
            normalized_aug_to_orig[aug_idx] = orig_idx
            if orig_to_aug[orig_idx] is None:
                orig_to_aug[orig_idx] = aug_idx

    return {
        "orig_to_aug": orig_to_aug,
        "aug_to_orig": normalized_aug_to_orig,
        "opcodes": [],
    }


def _validate_parallel_examples(
    original_sequences: List[List[str]],
    augmented_sequences: List[List[str]],
    error_message: str,
) -> None:
    if len(original_sequences) != len(augmented_sequences):
        logger.error(error_message)
        raise ValueError(error_message)


def span_corruption_rate(
    original_sentences: List[List[str]],
    original_labels: List[List[str]],
    augmented_sentences: List[List[str]],
    augmented_labels: List[List[str]],
    alignment_mapping: List[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute entity span corruption caused by augmentation.

    A source entity span is counted as corrupted when its original tokens are no
    longer recoverable as one contiguous augmented entity of the same type. This
    catches deleted, changed, split, merged, shifted-label, and type-changed
    entity spans. Non-entity edits outside the span do not count as span
    corruption.

    The primary `span_corruption_rate` is span-level:
    corrupted original entity spans / total original entity spans.
    `sentence_corruption_rate` is also returned for the older sentence-level
    interpretation used by the notebook printout.
    """
    logger.info(
        "Starting span corruption analysis on %s sentences",
        len(augmented_sentences),
    )
    _validate_parallel_examples(
        original_sentences,
        augmented_sentences,
        "Original and augmented sentences must have same length",
    )
    _validate_parallel_examples(
        original_labels,
        augmented_labels,
        "Original and augmented labels must have same length",
    )

    corrupted_sentences = 0
    corrupted_spans = 0
    total_spans = 0
    deleted_spans = 0
    split_spans = 0
    merged_or_expanded_spans = 0
    shifted_or_type_changed_spans = 0
    text_changed_spans = 0

    alignments = (
        alignment_mapping
        if alignment_mapping is not None
        else [None] * len(original_sentences)
    )

    for idx, (orig_sent, orig_seq, aug_sent, aug_seq, alignment) in enumerate(
        zip(
            original_sentences,
            original_labels,
            augmented_sentences,
            augmented_labels,
            alignments,
        )
    ):
        orig_entities = extract_entities_from_labels(orig_sent, orig_seq)
        aug_entities = extract_entities_from_labels(aug_sent, aug_seq)
        aug_entity_by_index = {}
        for aug_entity in aug_entities:
            aug_start, aug_end, _ = aug_entity
            for aug_idx in range(aug_start, aug_end + 1):
                aug_entity_by_index[aug_idx] = aug_entity

        if alignment is None:
            token_alignment = align_original_and_augmented_tokens(orig_sent, aug_sent)
        else:
            token_alignment = _alignment_from_aug_to_orig(
                len(orig_sent), len(aug_sent), alignment
            )
        orig_to_aug = token_alignment["orig_to_aug"]

        sentence_has_corruption = False

        for start, end, entity_type in orig_entities:
            total_spans += 1
            orig_indices = list(range(start, end + 1))
            aligned_aug_indices = [orig_to_aug[i] for i in orig_indices]
            present_aug_indices = [i for i in aligned_aug_indices if i is not None]

            reason = None
            if len(present_aug_indices) != len(orig_indices):
                reason = "deleted"
                deleted_spans += 1
            elif present_aug_indices != list(
                range(present_aug_indices[0], present_aug_indices[-1] + 1)
            ):
                reason = "split"
                split_spans += 1
            else:
                aug_start = present_aug_indices[0]
                aug_end = present_aug_indices[-1]
                aug_span_entities = {
                    aug_entity_by_index.get(aug_idx)
                    for aug_idx in range(aug_start, aug_end + 1)
                }
                aug_span_entities.discard(None)

                if len(aug_span_entities) != 1:
                    reason = "split_or_missing_label"
                    split_spans += 1
                else:
                    aug_entity = next(iter(aug_span_entities))
                    aug_entity_start, aug_entity_end, aug_entity_type = aug_entity

                    if (
                        aug_entity_start != aug_start
                        or aug_entity_end != aug_end
                    ):
                        reason = "merged_or_expanded"
                        merged_or_expanded_spans += 1
                    elif aug_entity_type != entity_type:
                        reason = "type_changed"
                        shifted_or_type_changed_spans += 1
                    else:
                        orig_span_tokens = orig_sent[start : end + 1]
                        aug_span_tokens = aug_sent[aug_start : aug_end + 1]
                        if orig_span_tokens != aug_span_tokens:
                            reason = "text_changed"
                            text_changed_spans += 1

            if reason is not None:
                corrupted_spans += 1
                sentence_has_corruption = True
                logger.debug(
                    "Sentence %s: corrupted span %s-%s (%s), reason=%s",
                    idx,
                    start,
                    end,
                    entity_type,
                    reason,
                )

        if sentence_has_corruption:
            corrupted_sentences += 1

    total_augmented = len(augmented_sentences)
    span_rate = corrupted_spans / total_spans if total_spans > 0 else 0.0
    sentence_rate = (
        corrupted_sentences / total_augmented if total_augmented > 0 else 0.0
    )

    logger.info(
        "Span corruption analysis complete: %s/%s spans corrupted (%.2f%%)",
        corrupted_spans,
        total_spans,
        span_rate * 100,
    )

    return {
        "span_corruption_rate": span_rate,
        "sentence_corruption_rate": sentence_rate,
        "total_augmented_sentences": total_augmented,
        "corrupted_sentences": corrupted_sentences,
        "total_spans": total_spans,
        "corrupted_spans": corrupted_spans,
        "deleted_spans": deleted_spans,
        "split_spans": split_spans,
        "merged_or_expanded_spans": merged_or_expanded_spans,
        "shifted_or_type_changed_spans": shifted_or_type_changed_spans,
        "text_changed_spans": text_changed_spans,
    }


def label_flip_rate(
    original_labels: List[List[str]],
    augmented_labels: List[List[str]],
    alignment_mapping: List[List[int]] = None,
    original_sentences: Optional[List[List[str]]] = None,
    augmented_sentences: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Compute label flips and broader label inconsistencies.

    `label_flip_rate` preserves the original project meaning: entity <-> O
    changes. `label_inconsistency_rate` additionally counts entity type changes
    such as PER -> ORG. When token sequences are provided, insertions/deletions
    are handled through token alignment instead of naive same-index comparison.
    """
    logger.info("Starting label flip analysis on %s sequences", len(augmented_labels))
    _validate_parallel_examples(
        original_labels,
        augmented_labels,
        "Original and augmented labels must have same length",
    )

    if (original_sentences is None) != (augmented_sentences is None):
        raise ValueError(
            "original_sentences and augmented_sentences must be provided together"
        )
    if original_sentences is not None:
        _validate_parallel_examples(
            original_sentences,
            augmented_sentences,
            "Original and augmented sentences must have same length",
        )

    total_tokens = 0
    flipped_tokens = 0
    inconsistent_tokens = 0
    entity_to_non_entity = 0
    non_entity_to_entity = 0
    label_type_changes = 0
    deleted_entity_tokens = 0
    inserted_entity_tokens = 0

    alignments = (
        alignment_mapping
        if alignment_mapping is not None
        else [None] * len(original_labels)
    )

    for example_idx, (orig_seq, aug_seq, alignment) in enumerate(
        zip(original_labels, augmented_labels, alignments)
    ):
        if original_sentences is not None:
            token_alignment = align_original_and_augmented_tokens(
                original_sentences[example_idx],
                augmented_sentences[example_idx],
            )
        else:
            token_alignment = _alignment_from_aug_to_orig(
                len(orig_seq),
                len(aug_seq),
                alignment,
            )

        orig_to_aug = token_alignment["orig_to_aug"]
        aug_to_orig = token_alignment["aug_to_orig"]

        for orig_idx, orig_label in enumerate(orig_seq):
            total_tokens += 1
            orig_type = normalize_entity_type(orig_label)
            aug_idx = orig_to_aug[orig_idx] if orig_idx < len(orig_to_aug) else None
            aug_label = aug_seq[aug_idx] if aug_idx is not None and aug_idx < len(aug_seq) else "O"
            aug_type = normalize_entity_type(aug_label)

            if orig_type != aug_type:
                inconsistent_tokens += 1
                if orig_type is not None and aug_type is None:
                    flipped_tokens += 1
                    entity_to_non_entity += 1
                    if aug_idx is None:
                        deleted_entity_tokens += 1
                elif orig_type is None and aug_type is not None:
                    flipped_tokens += 1
                    non_entity_to_entity += 1
                elif orig_type is not None and aug_type is not None:
                    label_type_changes += 1

        for aug_idx, orig_idx in enumerate(aug_to_orig):
            if orig_idx is None:
                aug_type = normalize_entity_type(aug_seq[aug_idx])
                if aug_type is not None:
                    total_tokens += 1
                    flipped_tokens += 1
                    inconsistent_tokens += 1
                    non_entity_to_entity += 1
                    inserted_entity_tokens += 1

    flip_rate = flipped_tokens / total_tokens if total_tokens > 0 else 0.0
    inconsistency_rate = (
        inconsistent_tokens / total_tokens if total_tokens > 0 else 0.0
    )

    logger.info(
        "Label flip analysis complete: %s/%s entity/O flips (%.2f%%), %s type changes",
        flipped_tokens,
        total_tokens,
        flip_rate * 100,
        label_type_changes,
    )

    return {
        "label_flip_rate": flip_rate,
        "label_inconsistency_rate": inconsistency_rate,
        "total_tokens": total_tokens,
        "flipped_tokens": flipped_tokens,
        "inconsistent_tokens": inconsistent_tokens,
        "entity_to_non_entity": entity_to_non_entity,
        "non_entity_to_entity": non_entity_to_entity,
        "label_type_changes": label_type_changes,
        "deleted_entity_tokens": deleted_entity_tokens,
        "inserted_entity_tokens": inserted_entity_tokens,
    }


def compute_entity_token_perturbation(
    original_tokens: List[str],
    original_labels: List[str],
    augmented_tokens: List[str],
    augmented_labels: List[str],
) -> Dict[str, float]:
    """
    Measure whether augmentation changed entity-bearing tokens more than O tokens.

    A source token is perturbed if it is deleted or aligned to a different token.
    Inserted augmented tokens are included in the all-token rate and counted
    separately by whether the inserted token carries an entity label.
    """
    alignment = align_original_and_augmented_tokens(original_tokens, augmented_tokens)
    orig_to_aug = alignment["orig_to_aug"]
    aug_to_orig = alignment["aug_to_orig"]

    original_token_changes = 0
    entity_token_changes = 0
    non_entity_token_changes = 0
    total_entity_tokens = 0
    total_non_entity_tokens = 0

    for orig_idx, token in enumerate(original_tokens):
        label = original_labels[orig_idx] if orig_idx < len(original_labels) else "O"
        is_entity = normalize_entity_type(label) is not None
        if is_entity:
            total_entity_tokens += 1
        else:
            total_non_entity_tokens += 1

        aug_idx = orig_to_aug[orig_idx]
        changed = aug_idx is None or aug_idx >= len(augmented_tokens)
        if not changed:
            changed = token != augmented_tokens[aug_idx]

        if changed:
            original_token_changes += 1
            if is_entity:
                entity_token_changes += 1
            else:
                non_entity_token_changes += 1

    inserted_tokens = sum(1 for orig_idx in aug_to_orig if orig_idx is None)
    inserted_entity_tokens = sum(
        1
        for aug_idx, orig_idx in enumerate(aug_to_orig)
        if orig_idx is None
        and aug_idx < len(augmented_labels)
        and normalize_entity_type(augmented_labels[aug_idx]) is not None
    )
    inserted_non_entity_tokens = inserted_tokens - inserted_entity_tokens

    all_denominator = len(original_tokens) + inserted_tokens
    all_perturbed = original_token_changes + inserted_tokens

    return {
        "all_token_perturbation_rate": (
            all_perturbed / all_denominator if all_denominator > 0 else 0.0
        ),
        "entity_token_perturbation_rate": (
            entity_token_changes / total_entity_tokens
            if total_entity_tokens > 0
            else 0.0
        ),
        "non_entity_token_perturbation_rate": (
            non_entity_token_changes / total_non_entity_tokens
            if total_non_entity_tokens > 0
            else 0.0
        ),
        "total_original_tokens": len(original_tokens),
        "total_entity_tokens": total_entity_tokens,
        "total_non_entity_tokens": total_non_entity_tokens,
        "changed_or_deleted_original_tokens": original_token_changes,
        "changed_or_deleted_entity_tokens": entity_token_changes,
        "changed_or_deleted_non_entity_tokens": non_entity_token_changes,
        "inserted_tokens": inserted_tokens,
        "inserted_entity_tokens": inserted_entity_tokens,
        "inserted_non_entity_tokens": inserted_non_entity_tokens,
    }


def entity_token_perturbation_rate(
    original_sentences: List[List[str]],
    original_labels: List[List[str]],
    augmented_sentences: List[List[str]],
    augmented_labels: List[List[str]],
) -> Dict[str, float]:
    """Aggregate entity/non-entity token perturbation over a dataset."""
    _validate_parallel_examples(
        original_sentences,
        augmented_sentences,
        "Original and augmented sentences must have same length",
    )
    _validate_parallel_examples(
        original_labels,
        augmented_labels,
        "Original and augmented labels must have same length",
    )

    totals = {
        "total_original_tokens": 0,
        "total_entity_tokens": 0,
        "total_non_entity_tokens": 0,
        "changed_or_deleted_original_tokens": 0,
        "changed_or_deleted_entity_tokens": 0,
        "changed_or_deleted_non_entity_tokens": 0,
        "inserted_tokens": 0,
        "inserted_entity_tokens": 0,
        "inserted_non_entity_tokens": 0,
    }

    for orig_tokens, orig_seq, aug_tokens, aug_seq in zip(
        original_sentences,
        original_labels,
        augmented_sentences,
        augmented_labels,
    ):
        result = compute_entity_token_perturbation(
            orig_tokens,
            orig_seq,
            aug_tokens,
            aug_seq,
        )
        for key in totals:
            totals[key] += result[key]

    all_denominator = totals["total_original_tokens"] + totals["inserted_tokens"]
    all_perturbed = (
        totals["changed_or_deleted_original_tokens"] + totals["inserted_tokens"]
    )

    return {
        "all_token_perturbation_rate": (
            all_perturbed / all_denominator if all_denominator > 0 else 0.0
        ),
        "entity_token_perturbation_rate": (
            totals["changed_or_deleted_entity_tokens"] / totals["total_entity_tokens"]
            if totals["total_entity_tokens"] > 0
            else 0.0
        ),
        "non_entity_token_perturbation_rate": (
            totals["changed_or_deleted_non_entity_tokens"]
            / totals["total_non_entity_tokens"]
            if totals["total_non_entity_tokens"] > 0
            else 0.0
        ),
        **totals,
    }


def per_type_f1_scores(
    y_true: List[List[str]], y_pred: List[List[str]], entity_types: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute entity-level F1, precision, and recall for each entity type.

    Uses seqeval, so the scores are span-level NER metrics rather than token
    classification accuracy. Support is the number of true entity spans of that
    type, not the number of entity-labeled tokens.
    """
    logger.info("Computing per-type F1 scores for %s sequences", len(y_true))
    _validate_parallel_examples(
        y_true,
        y_pred,
        "True and predicted label sequences must have same length",
    )

    if entity_types is None:
        entity_types_set = set()
        for seq in y_true + y_pred:
            for label in seq:
                entity_type = normalize_entity_type(label)
                if entity_type is not None:
                    entity_types_set.add(entity_type)
        entity_types = sorted(entity_types_set)

    results: Dict[str, Dict[str, float]] = {}

    overall_precision = precision_score(y_true, y_pred, zero_division=0)
    overall_recall = recall_score(y_true, y_pred, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)
    overall_support = sum(
        len(extract_entities_from_labels([""] * len(seq), seq)) for seq in y_true
    )

    results["overall"] = {
        "f1": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
        "support": overall_support,
    }

    for entity_type in entity_types:
        y_true_type = []
        y_pred_type = []

        for true_seq, pred_seq in zip(y_true, y_pred):
            true_seq_filtered = []
            pred_seq_filtered = []

            for true_label, pred_label in zip(true_seq, pred_seq):
                if normalize_entity_type(true_label) == entity_type:
                    true_seq_filtered.append(true_label)
                else:
                    true_seq_filtered.append("O")

                if normalize_entity_type(pred_label) == entity_type:
                    pred_seq_filtered.append(pred_label)
                else:
                    pred_seq_filtered.append("O")

            y_true_type.append(true_seq_filtered)
            y_pred_type.append(pred_seq_filtered)

        precision = precision_score(y_true_type, y_pred_type, zero_division=0)
        recall = recall_score(y_true_type, y_pred_type, zero_division=0)
        f1 = f1_score(y_true_type, y_pred_type, zero_division=0)
        support = sum(
            1
            for seq in y_true
            for _, _, true_type in extract_entities_from_labels([""] * len(seq), seq)
            if true_type == entity_type
        )

        logger.info(
            "%s: F1=%.4f, Precision=%.4f, Recall=%.4f, Support=%s",
            entity_type,
            f1,
            precision,
            recall,
            support,
        )

        results[entity_type] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": support,
        }

    return results


def compute_f1_by_entity_type(
    predictions: List[List[str]],
    references: List[List[str]],
    entity_types: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper using the prediction/reference naming from reports."""
    return per_type_f1_scores(references, predictions, entity_types)


def aggregate_metrics_by_augmentation_method(results: Dict) -> Dict[str, Dict]:
    """
    Extract a compact method -> metrics table from saved experiment JSON content.
    """
    models = results.get("models", results)
    summary = {}
    for method, payload in models.items():
        evaluation = payload.get("evaluation", {})
        quality = payload.get("augmentation_quality", {})
        summary[method] = {
            "eval_f1": evaluation.get("eval_f1"),
            "eval_precision": evaluation.get("eval_precision"),
            "eval_recall": evaluation.get("eval_recall"),
            "span_corruption_rate": quality.get("span_corruption", {}).get(
                "span_corruption_rate"
            ),
            "label_flip_rate": quality.get("label_flip", {}).get("label_flip_rate"),
            "label_inconsistency_rate": quality.get("label_flip", {}).get(
                "label_inconsistency_rate"
            ),
            "per_type_f1_scores": payload.get("per_type_f1_scores", {}),
        }
    return summary


def augmentation_quality_report(
    original_sentences: List[List[str]],
    original_labels: List[List[str]],
    augmented_sentences: List[List[str]],
    augmented_labels: List[List[str]],
    predictions: List[List[str]] = None,
    entity_types: List[str] = None,
) -> Dict:
    """
    Generate a combined report for augmentation quality and optional model F1.
    """
    logger.info("=" * 60)
    logger.info("Generating comprehensive augmentation quality report")
    logger.info("Processing %s augmented samples", len(augmented_sentences))
    logger.info("=" * 60)

    report = {
        "span_corruption": span_corruption_rate(
            original_sentences,
            original_labels,
            augmented_sentences,
            augmented_labels,
        ),
        "label_flip": label_flip_rate(
            original_labels,
            augmented_labels,
            original_sentences=original_sentences,
            augmented_sentences=augmented_sentences,
        ),
        "entity_token_perturbation": entity_token_perturbation_rate(
            original_sentences,
            original_labels,
            augmented_sentences,
            augmented_labels,
        ),
    }

    if predictions is not None:
        logger.info("Computing per-type F1 scores...")
        report["per_type_f1"] = per_type_f1_scores(
            augmented_labels,
            predictions,
            entity_types,
        )

    logger.info("Augmentation quality report complete")
    logger.info("=" * 60)

    return report
