"""
Comprehensive test suite for metrics.py using pytest.
Tests the evaluation metrics for augmented NER datasets.
"""

import sys
import os
from pathlib import Path
import pytest
import logging

# Add the metrics directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "metrics"))

from metrics import (
    extract_entities_from_labels,
    span_corruption_rate,
    label_flip_rate,
    per_type_f1_scores,
    augmentation_quality_report,
)

# Get logger for tests (pytest will handle configuration)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_ner_data():
    """
    Create sample NER data for testing metrics.

    Returns:
        A dictionary containing original and augmented sentences with labels.
    """
    # Simple original sentences with clear entity patterns
    original_sentences = [
        ["John", "Smith", "works", "for", "Google"],
        ["Paris", "is", "in", "France"],
        ["The", "CEO", "of", "Microsoft", "is", "Satya"],
    ]

    original_labels = [
        ["B-PER", "I-PER", "O", "O", "B-ORG"],
        ["B-LOC", "O", "O", "B-LOC"],
        ["O", "O", "O", "B-ORG", "O", "B-PER"],
    ]

    return {
        "original_sentences": original_sentences,
        "original_labels": original_labels,
    }


@pytest.fixture
def augmented_no_corruption(sample_ner_data):
    """
    Create augmented data with NO corruption (perfect augmentation).
    """
    original = sample_ner_data

    # Identical to original - no corruption at all
    augmented_sentences = [sent.copy() for sent in original["original_sentences"]]
    augmented_labels = [labels.copy() for labels in original["original_labels"]]

    return {
        "augmented_sentences": augmented_sentences,
        "augmented_labels": augmented_labels,
    }


@pytest.fixture
def augmented_with_span_corruption(sample_ner_data):
    """
    Create augmented data with span-level corruption (entity spans lost/changed).
    """
    original = sample_ner_data

    # First sentence: entity span "John Smith" -> "John" (partial corruption)
    augmented_sentences = [
        ["John", "works", "for", "Google"],  # "Smith" removed
        ["Paris", "is", "in", "France"],  # No corruption
        ["The", "CEO", "of", "Microsoft", "is", "Satya"],  # No corruption
    ]

    augmented_labels = [
        ["B-PER", "O", "O", "B-ORG"],  # "Smith" removed, label lost
        ["B-LOC", "O", "O", "B-LOC"],
        ["O", "O", "O", "B-ORG", "O", "B-PER"],
    ]

    return {
        "augmented_sentences": augmented_sentences,
        "augmented_labels": augmented_labels,
    }


@pytest.fixture
def augmented_with_label_flips(sample_ner_data):
    """
    Create augmented data with label-level flips (entity <-> non-entity changes).
    """
    original = sample_ner_data

    # Introduce label flips: entities become non-entities and vice versa
    augmented_sentences = [
        ["John", "Smith", "works", "for", "Google"],  # Same tokens
        ["Paris", "is", "in", "France"],  # Same tokens
        ["The", "CEO", "of", "Microsoft", "is", "Satya"],  # Same tokens
    ]

    augmented_labels = [
        ["O", "O", "O", "O", "O"],  # All entities flipped to O
        ["B-LOC", "O", "O", "B-LOC"],  # No change
        ["O", "B-PER", "O", "O", "O", "O"],  # Mixed flips
    ]

    return {
        "augmented_sentences": augmented_sentences,
        "augmented_labels": augmented_labels,
    }


@pytest.fixture
def predictions_good():
    """Create good predictions (high F1 score)."""
    predictions = [
        ["B-PER", "I-PER", "O", "O", "B-ORG"],
        ["B-LOC", "O", "O", "B-LOC"],
        ["O", "O", "O", "B-ORG", "O", "B-PER"],
    ]
    return predictions


@pytest.fixture
def predictions_poor():
    """Create poor predictions (low F1 score)."""
    predictions = [
        ["O", "O", "B-PER", "B-ORG", "B-ORG"],
        ["O", "B-LOC", "B-LOC", "O"],
        ["B-MISC", "B-MISC", "B-PER", "O", "O", "O"],
    ]
    return predictions


@pytest.fixture
def predictions_poor_augmented():
    """Create poor predictions matching augmented_with_span_corruption structure (4, 4, 6 tokens)."""
    predictions = [
        ["O", "B-PER", "B-ORG", "B-ORG"],  # 4 tokens (matches augmented first sentence)
        ["O", "B-LOC", "B-LOC", "O"],  # 4 tokens
        ["B-MISC", "B-MISC", "B-PER", "O", "O", "O"],  # 6 tokens
    ]
    return predictions


class TestExtractEntitiesFromLabels:
    """Tests for extract_entities_from_labels function."""

    def test_extract_simple_entities(self):
        """Test extracting entities from simple BIO sequence."""
        tokens = ["John", "Smith", "works"]
        labels = ["B-PER", "I-PER", "O"]

        entities = extract_entities_from_labels(tokens, labels)

        assert len(entities) == 1, "Should extract 1 entity"
        assert entities[0] == (
            0,
            1,
            "PER",
        ), "Should identify John Smith as PER from indices 0-1"

    def test_extract_multiple_entities(self):
        """Test extracting multiple entities."""
        tokens = ["John", "Smith", "works", "for", "Google"]
        labels = ["B-PER", "I-PER", "O", "O", "B-ORG"]

        entities = extract_entities_from_labels(tokens, labels)

        assert len(entities) == 2, "Should extract 2 entities"
        assert (0, 1, "PER") in entities, "Should find John Smith (PER)"
        assert (4, 4, "ORG") in entities, "Should find Google (ORG)"

    def test_extract_no_entities(self):
        """Test extracting from sequence with no entities."""
        tokens = ["The", "text", "has", "no", "entities"]
        labels = ["O", "O", "O", "O", "O"]

        entities = extract_entities_from_labels(tokens, labels)

        assert len(entities) == 0, "Should extract 0 entities"

    def test_extract_consecutive_different_entities(self):
        """Test entities with different types in sequence."""
        tokens = ["Paris", "is", "in", "France"]
        labels = ["B-LOC", "O", "O", "B-LOC"]

        entities = extract_entities_from_labels(tokens, labels)

        assert len(entities) == 2, "Should extract 2 separate entities"
        assert (0, 0, "LOC") in entities, "Should find Paris"
        assert (3, 3, "LOC") in entities, "Should find France"

    def test_extract_multiple_entity_types(self):
        """Test extracting multiple entity types."""
        tokens = ["John", "works", "for", "Google", "in", "California"]
        labels = ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"]

        entities = extract_entities_from_labels(tokens, labels)

        assert len(entities) == 3, "Should extract 3 entities"
        assert any(e[2] == "PER" for e in entities), "Should find PER entity"
        assert any(e[2] == "ORG" for e in entities), "Should find ORG entity"
        assert any(e[2] == "LOC" for e in entities), "Should find LOC entity"

    def test_extract_handles_invalid_transitions(self):
        """Test handling of I tag without preceding B tag."""
        tokens = ["Word1", "Word2", "Word3"]
        labels = ["I-PER", "I-PER", "O"]

        entities = extract_entities_from_labels(tokens, labels)

        # Should handle gracefully (treat as entity start)
        assert len(entities) >= 1, "Should handle I without B tag"


class TestSpanCorruptionRate:
    """Tests for span_corruption_rate function."""

    def test_no_corruption(self, sample_ner_data, augmented_no_corruption):
        """Test span corruption rate when there is no corruption."""
        result = span_corruption_rate(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_sentences"],
            augmented_no_corruption["augmented_labels"],
        )

        assert result["span_corruption_rate"] == 0.0, "Should be 0 corruption"
        assert result["corrupted_sentences"] == 0, "Should have no corrupted sentences"
        assert result["corrupted_spans"] == 0, "Should have no corrupted spans"

    def test_with_corruption(self, sample_ner_data, augmented_with_span_corruption):
        """Test span corruption rate with corrupted spans."""
        result = span_corruption_rate(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_with_span_corruption["augmented_sentences"],
            augmented_with_span_corruption["augmented_labels"],
        )

        assert result["span_corruption_rate"] > 0, "Should detect corruption"
        assert result["corrupted_sentences"] > 0, "Should find corrupted sentences"
        assert result["corrupted_spans"] > 0, "Should find corrupted spans"

    def test_partial_corruption(self, sample_ner_data, augmented_with_span_corruption):
        """Test that partial corruption is detected correctly."""
        result = span_corruption_rate(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_with_span_corruption["augmented_sentences"],
            augmented_with_span_corruption["augmented_labels"],
        )

        # First sentence has corruption, but 2nd and 3rd don't
        assert result["corrupted_sentences"] == 1, "Should have 1 corrupted sentence"
        assert result["total_augmented_sentences"] == 3, "Should have 3 total sentences"
        corruption_rate = result["span_corruption_rate"]
        assert 0.0 < corruption_rate < 1.0, "Corruption rate should be between 0 and 1"

    def test_returns_required_keys(self, sample_ner_data, augmented_no_corruption):
        """Test that result has all required keys."""
        result = span_corruption_rate(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_sentences"],
            augmented_no_corruption["augmented_labels"],
        )

        required_keys = {
            "span_corruption_rate",
            "total_augmented_sentences",
            "corrupted_sentences",
            "total_spans",
            "corrupted_spans",
        }
        assert required_keys.issubset(
            result.keys()
        ), f"Result should have keys: {required_keys}"

    def test_mismatched_lengths_raises_error(self, sample_ner_data):
        """Test that mismatched input lengths raise ValueError."""
        original_sents = sample_ner_data["original_sentences"]
        original_labels = sample_ner_data["original_labels"]

        # Provide only 2 augmented sentences instead of 3
        mismatched_aug_sents = [["John", "Smith"]]
        mismatched_aug_labels = [["B-PER", "I-PER"]]

        with pytest.raises(
            ValueError, match="Original and augmented sentences must have same length"
        ):
            span_corruption_rate(
                original_sents,
                original_labels,
                mismatched_aug_sents,
                mismatched_aug_labels,
            )


class TestLabelFlipRate:
    """Tests for label_flip_rate function."""

    def test_no_label_flips(self, sample_ner_data, augmented_no_corruption):
        """Test label flip rate when there are no flips."""
        result = label_flip_rate(
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_labels"],
        )

        assert result["label_flip_rate"] == 0.0, "Should have 0 label flips"
        assert result["flipped_tokens"] == 0, "Should have 0 flipped tokens"

    def test_with_label_flips(self, sample_ner_data, augmented_with_label_flips):
        """Test label flip rate with flipped labels."""
        result = label_flip_rate(
            sample_ner_data["original_labels"],
            augmented_with_label_flips["augmented_labels"],
        )

        assert result["label_flip_rate"] > 0, "Should detect label flips"
        assert result["flipped_tokens"] > 0, "Should count flipped tokens"

    def test_entity_to_non_entity_flips(self, sample_ner_data):
        """Test counting entity->non-entity flips."""
        original = [["B-PER", "I-PER", "O"]]
        augmented = [["O", "O", "O"]]  # All entities become O

        result = label_flip_rate(original, augmented)

        assert result["entity_to_non_entity"] == 2, "Should count 2 entity->O flips"
        assert result["non_entity_to_entity"] == 0, "Should count 0 O->entity flips"
        assert result["flipped_tokens"] == 2, "Should count 2 total flips"

    def test_non_entity_to_entity_flips(self, sample_ner_data):
        """Test counting non-entity->entity flips."""
        original = [["O", "O", "O"]]
        augmented = [["B-PER", "I-PER", "B-LOC"]]  # All O become entities

        result = label_flip_rate(original, augmented)

        assert result["non_entity_to_entity"] == 3, "Should count 3 O->entity flips"
        assert result["entity_to_non_entity"] == 0, "Should count 0 entity->O flips"
        assert result["flipped_tokens"] == 3, "Should count 3 total flips"

    def test_mixed_label_flips(self):
        """Test mixed entity flips (both directions)."""
        original = [["B-PER", "I-PER", "O", "B-ORG"]]
        augmented = [["O", "O", "B-LOC", "O"]]

        result = label_flip_rate(original, augmented)

        assert (
            result["entity_to_non_entity"] == 3
        ), "Should have 3 entity->O flips (B-PER, I-PER, B-ORG)"
        assert (
            result["non_entity_to_entity"] == 1
        ), "Should have 1 O->entity flip (B-LOC)"
        assert result["flipped_tokens"] == 4, "Should have 4 total flips"

    def test_label_type_changes_not_counted(self):
        """Test that changing from one entity type to another is not counted as flip."""
        original = [["B-PER", "O"]]
        augmented = [["B-ORG", "O"]]  # Changed type but still entity

        result = label_flip_rate(original, augmented)

        assert (
            result["flipped_tokens"] == 0
        ), "Type change should not count as flip (stays entity)"
        assert result["total_tokens"] == 2, "Should count total tokens"
        assert result["label_flip_rate"] == 0.0, "Type change should not count as flip"

    def test_returns_required_keys(self, sample_ner_data, augmented_no_corruption):
        """Test that result has all required keys."""
        result = label_flip_rate(
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_labels"],
        )

        required_keys = {
            "label_flip_rate",
            "total_tokens",
            "flipped_tokens",
            "entity_to_non_entity",
            "non_entity_to_entity",
        }
        assert required_keys.issubset(
            result.keys()
        ), f"Result should have keys: {required_keys}"

    def test_mismatched_lengths_raises_error(self, sample_ner_data):
        """Test that mismatched input lengths raise ValueError."""
        original_labels = sample_ner_data["original_labels"]
        mismatched_aug_labels = [["B-PER", "I-PER"]]

        with pytest.raises(
            ValueError, match="Original and augmented labels must have same length"
        ):
            label_flip_rate(original_labels, mismatched_aug_labels)


class TestPerTypeF1Scores:
    """Tests for per_type_f1_scores function."""

    def test_perfect_predictions(self, sample_ner_data, predictions_good):
        """Test F1 scores with perfect predictions."""
        result = per_type_f1_scores(
            sample_ner_data["original_labels"], predictions_good
        )

        assert "overall" in result, "Should have overall scores"
        assert result["overall"]["f1"] == 1.0, "Perfect predictions should have F1=1.0"

    def test_poor_predictions(self, sample_ner_data, predictions_poor):
        """Test F1 scores with poor predictions."""
        result = per_type_f1_scores(
            sample_ner_data["original_labels"], predictions_poor
        )

        assert "overall" in result, "Should have overall scores"
        assert result["overall"]["f1"] < 1.0, "Poor predictions should have F1<1.0"

    def test_per_type_scores_exist(self, sample_ner_data, predictions_good):
        """Test that per-type scores are computed."""
        result = per_type_f1_scores(
            sample_ner_data["original_labels"], predictions_good
        )

        # Should have entries for each entity type in the data
        entity_types = {"PER", "ORG", "LOC"}
        for entity_type in entity_types:
            assert entity_type in result, f"Should have scores for {entity_type}"

    def test_per_type_has_required_keys(self, sample_ner_data, predictions_good):
        """Test that per-type results have required keys."""
        result = per_type_f1_scores(
            sample_ner_data["original_labels"], predictions_good
        )

        required_keys = {"f1", "precision", "recall"}
        for entity_type in ["PER", "ORG", "LOC"]:
            assert required_keys.issubset(
                result[entity_type].keys()
            ), f"{entity_type} should have {required_keys}"

    def test_overall_has_support(self, sample_ner_data, predictions_good):
        """Test that per-type results have support count."""
        result = per_type_f1_scores(
            sample_ner_data["original_labels"], predictions_good
        )

        for entity_type in ["PER", "ORG", "LOC"]:
            assert (
                "support" in result[entity_type]
            ), f"{entity_type} should have support"
            assert (
                result[entity_type]["support"] > 0
            ), f"{entity_type} should have positive support"

    def test_scores_between_0_and_1(self, sample_ner_data, predictions_poor):
        """Test that F1/precision/recall scores are between 0 and 1."""
        result = per_type_f1_scores(
            sample_ner_data["original_labels"], predictions_poor
        )

        for entity_type in result:
            if entity_type != "overall":
                f1 = result[entity_type]["f1"]
                precision = result[entity_type]["precision"]
                recall = result[entity_type]["recall"]

                assert 0.0 <= f1 <= 1.0, f"F1 should be between 0 and 1"
                assert 0.0 <= precision <= 1.0, f"Precision should be between 0 and 1"
                assert 0.0 <= recall <= 1.0, f"Recall should be between 0 and 1"

    def test_custom_entity_types(self, sample_ner_data, predictions_good):
        """Test specifying custom entity types."""
        entity_types = ["PER", "ORG"]
        result = per_type_f1_scores(
            sample_ner_data["original_labels"],
            predictions_good,
            entity_types=entity_types,
        )

        assert "PER" in result, "Should have PER scores"
        assert "ORG" in result, "Should have ORG scores"


class TestAugmentationQualityReport:
    """Tests for augmentation_quality_report function."""

    def test_basic_report_generation(self, sample_ner_data, augmented_no_corruption):
        """Test generating a basic augmentation quality report."""
        report = augmentation_quality_report(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_sentences"],
            augmented_no_corruption["augmented_labels"],
        )

        assert isinstance(report, dict), "Should return a dictionary"
        assert "span_corruption" in report, "Report should have span_corruption"
        assert "label_flip" in report, "Report should have label_flip"

    def test_report_without_predictions(self, sample_ner_data, augmented_no_corruption):
        """Test report without prediction data."""
        report = augmentation_quality_report(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_sentences"],
            augmented_no_corruption["augmented_labels"],
        )

        assert (
            "per_type_f1" not in report
        ), "Should not have per_type_f1 without predictions"

    def test_report_with_predictions(
        self, sample_ner_data, augmented_no_corruption, predictions_good
    ):
        """Test report with prediction data."""
        report = augmentation_quality_report(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_sentences"],
            augmented_no_corruption["augmented_labels"],
            predictions=predictions_good,
        )

        assert "per_type_f1" in report, "Should have per_type_f1 with predictions"
        assert "overall" in report["per_type_f1"], "Should have overall F1 scores"

    def test_report_combines_all_metrics(
        self,
        sample_ner_data,
        augmented_with_span_corruption,
        predictions_poor_augmented,
    ):
        """Test that report combines all three metric types."""
        report = augmentation_quality_report(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_with_span_corruption["augmented_sentences"],
            augmented_with_span_corruption["augmented_labels"],
            predictions=predictions_poor_augmented,
        )

        # Check all metric types are present
        assert "span_corruption" in report
        assert "label_flip" in report
        assert "per_type_f1" in report

        # Check structure of each metric
        assert isinstance(report["span_corruption"], dict)
        assert isinstance(report["label_flip"], dict)
        assert isinstance(report["per_type_f1"], dict)

    def test_report_with_custom_entity_types(
        self, sample_ner_data, augmented_no_corruption, predictions_good
    ):
        """Test report with custom entity types specified."""
        entity_types = ["PER", "ORG"]
        report = augmentation_quality_report(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_sentences"],
            augmented_no_corruption["augmented_labels"],
            predictions=predictions_good,
            entity_types=entity_types,
        )

        assert "PER" in report["per_type_f1"]
        assert "ORG" in report["per_type_f1"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        tokens = []
        labels = []

        entities = extract_entities_from_labels(tokens, labels)
        assert entities == [], "Should return empty list for empty input"

    def test_single_token_entity(self):
        """Test single-token entity extraction."""
        tokens = ["John"]
        labels = ["B-PER"]

        entities = extract_entities_from_labels(tokens, labels)
        assert len(entities) == 1, "Should extract single token entity"
        assert entities[0] == (0, 0, "PER"), "Should identify correct span"

    def test_all_b_tags_no_i_tags(self):
        """Test sequence with only B tags (no I tags)."""
        tokens = ["Word1", "Word2", "Word3"]
        labels = ["B-PER", "B-PER", "B-LOC"]

        entities = extract_entities_from_labels(tokens, labels)
        assert len(entities) == 3, "Should treat each B tag as separate entity"

    def test_very_long_span(self):
        """Test handling of very long entity spans."""
        tokens = [f"word_{i}" for i in range(100)]
        labels = ["B-PER"] + ["I-PER"] * 99

        entities = extract_entities_from_labels(tokens, labels)
        assert len(entities) == 1, "Should extract single long entity"
        assert entities[0] == (0, 99, "PER"), "Should get correct span boundaries"

    def test_span_corruption_all_corrupted(self, sample_ner_data):
        """Test when all spans are corrupted."""
        original_sents = sample_ner_data["original_sentences"]
        original_labels = sample_ner_data["original_labels"]

        # Create augmented data with all entities removed
        augmented_sents = [[word for word in sent] for sent in original_sents]
        augmented_labels = [["O"] * len(labels) for labels in original_labels]

        result = span_corruption_rate(
            original_sents, original_labels, augmented_sents, augmented_labels
        )

        # All sentences have corrupted spans
        assert result["span_corruption_rate"] == 1.0, "All should be corrupted"
        assert result["corrupted_sentences"] == len(original_sents)

    def test_label_flip_all_flipped(self):
        """Test when all labels are flipped."""
        original = [["B-PER", "I-PER", "B-ORG"]]
        augmented = [["O", "O", "O"]]

        result = label_flip_rate(original, augmented)

        total_tokens = len(original[0])
        assert result["flipped_tokens"] == total_tokens, "All tokens should be flipped"
        assert result["label_flip_rate"] == 1.0, "Flip rate should be 1.0"


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_evaluation_pipeline(
        self, sample_ner_data, augmented_with_span_corruption
    ):
        """Test complete evaluation pipeline."""
        # Step 1: Extract entities
        entities1 = extract_entities_from_labels(
            sample_ner_data["original_sentences"][0],
            sample_ner_data["original_labels"][0],
        )
        assert len(entities1) > 0, "Should extract entities from original"

        # Step 2: Compute span corruption
        corruption = span_corruption_rate(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_with_span_corruption["augmented_sentences"],
            augmented_with_span_corruption["augmented_labels"],
        )
        assert (
            corruption["span_corruption_rate"] >= 0
        ), "Corruption rate should be valid"

        # Step 3: Compute label flips
        flips = label_flip_rate(
            sample_ner_data["original_labels"],
            augmented_with_span_corruption["augmented_labels"],
        )
        assert flips["label_flip_rate"] >= 0, "Flip rate should be valid"

    def test_comparing_different_augmentations(
        self, sample_ner_data, augmented_no_corruption, augmented_with_span_corruption
    ):
        """Test comparing different augmentation strategies."""
        # Quality of no-corruption augmentation
        report1 = augmentation_quality_report(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_no_corruption["augmented_sentences"],
            augmented_no_corruption["augmented_labels"],
        )

        # Quality of span-corruption augmentation
        report2 = augmentation_quality_report(
            sample_ner_data["original_sentences"],
            sample_ner_data["original_labels"],
            augmented_with_span_corruption["augmented_sentences"],
            augmented_with_span_corruption["augmented_labels"],
        )

        # No-corruption should be better
        assert (
            report1["span_corruption"]["span_corruption_rate"]
            < report2["span_corruption"]["span_corruption_rate"]
        ), "No-corruption augmentation should have lower corruption rate"

    def test_metrics_consistency(self, sample_ner_data, augmented_no_corruption):
        """Test consistency of metrics across multiple runs."""
        # Run metrics multiple times
        results = []
        for _ in range(3):
            result = augmentation_quality_report(
                sample_ner_data["original_sentences"],
                sample_ner_data["original_labels"],
                augmented_no_corruption["augmented_sentences"],
                augmented_no_corruption["augmented_labels"],
            )
            results.append(result)

        # All runs should produce identical results (deterministic)
        for key in ["span_corruption", "label_flip"]:
            for metric_key in results[0][key]:
                values = [r[key][metric_key] for r in results]
                assert all(
                    v == values[0] for v in values
                ), f"{key}.{metric_key} should be consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
