# Boyan Metric Review Summary

## What the Project Measures

This project studies why data augmentation can fail for low-resource NER. Because NER is a token-level task, an augmentation can sound fluent while still damaging entity spans, BIO label boundaries, or entity labels. The core analysis compares model performance with data-level corruption metrics so the group can explain not only whether an augmentation helped, but why it helped or hurt.

## Metric Improvements Made

- Added robust BIO helpers for entity type normalization and entity span extraction.
- Added token alignment between original and augmented tokens using an edit-script heuristic, so insertions and deletions are not treated as simple same-index comparisons.
- Updated span corruption to report a span-level `span_corruption_rate` and a separate `sentence_corruption_rate`.
- Expanded span corruption diagnostics for deleted, split, merged/expanded, type-changed, and text-changed entity spans.
- Updated label flip logic to separate entity/O flips from broader label inconsistencies such as PER -> ORG.
- Added entity-token perturbation metrics for all tokens, entity tokens, and non-entity tokens.
- Fixed per-type F1 support to count true entity spans instead of entity-labeled tokens.
- Added convenience wrappers for `compute_f1_by_entity_type` and `aggregate_metrics_by_augmentation_method`.

## Important Notebook Caveat

`augmentation_quality_report(...)` now calls the improved alignment-aware metrics automatically. The current `driver.ipynb` metric cell still calls `label_flip_rate(original_labels, augmented_labels)` directly, which cannot use token alignment unless original and augmented token lists are also passed. For the most reliable notebook output, update that call to include:

```python
label_flip = label_flip_rate(
    original_labels=original_train_ner,
    augmented_labels=augmented_train_ner,
    original_sentences=original_train_tokens,
    augmented_sentences=augmented_train_tokens,
)
```

To include the new token perturbation metric in the notebook, import and call:

```python
from metrics import entity_token_perturbation_rate

token_perturbation = entity_token_perturbation_rate(
    original_train_tokens,
    original_train_ner,
    augmented_train_tokens,
    augmented_train_ner,
)
```

## Verification

The focused metrics test suite passes in the `musesti` environment:

```bash
"C:\Users\Boyan - Durr lab\conda-envs\musesti\python.exe" -m pytest src/test/test_metrics.py -q
```

Result:

```text
50 passed
```

## Presentation Takeaway

The main story is that augmentation quality matters as much as augmentation quantity. Methods that preserve entity spans and labels are safer for low-resource NER, while methods that delete, split, shift, or relabel entity-bearing tokens can reduce F1 even if they increase surface-level text diversity.
