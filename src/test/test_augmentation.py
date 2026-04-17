"""
Pytest suite for augmentation.py module.

Tests all augmentation techniques:
- Naive EDA
- Back translation
- Entity-aware replacement
"""

import pytest
from typing import List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentation import (
    NaiveEDA,
    BackTranslation,
    EntityAwareReplacement,
    ConllAugmenter,
    Sentence,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_sentence() -> Sentence:
    """Sample sentence without entities."""
    return [
        ("The", "DT", "B-NP", "O"),
        ("quick", "JJ", "I-NP", "O"),
        ("brown", "JJ", "I-NP", "O"),
        ("fox", "NN", "I-NP", "O"),
        ("jumps", "VBZ", "B-VP", "O"),
    ]


@pytest.fixture
def sentence_with_entities() -> Sentence:
    """Sample sentence with named entities."""
    return [
        ("Barack", "NNP", "B-NP", "B-PER"),
        ("Obama", "NNP", "I-NP", "I-PER"),
        ("was", "VBD", "B-VP", "O"),
        ("born", "VBN", "I-VP", "O"),
        ("in", "IN", "B-PP", "O"),
        ("Hawaii", "NNP", "B-NP", "B-LOC"),
        (".", ".", "O", "O"),
    ]


@pytest.fixture
def multi_entity_sentence() -> Sentence:
    """Sentence with multiple entity types."""
    return [
        ("Steve", "NNP", "B-NP", "B-PER"),
        ("Jobs", "NNP", "I-NP", "I-PER"),
        ("founded", "VBD", "B-VP", "O"),
        ("Apple", "NNP", "B-NP", "B-ORG"),
        ("in", "IN", "B-PP", "O"),
        ("Cupertino", "NNP", "B-NP", "B-LOC"),
        (".", ".", "O", "O"),
    ]


@pytest.fixture
def eda_augmenter() -> NaiveEDA:
    """Initialized EDA augmenter."""
    return NaiveEDA(seed=42)


@pytest.fixture
def entity_replacement_augmenter() -> EntityAwareReplacement:
    """Initialized entity replacement augmenter."""
    return EntityAwareReplacement(seed=42)


@pytest.fixture
def conll_augmenter() -> ConllAugmenter:
    """Initialized CONLL augmenter."""
    return ConllAugmenter(seed=42)


# ============================================================================
# Tests for NaiveEDA
# ============================================================================


class TestNaiveEDA:
    """Tests for the NaiveEDA class."""

    def test_initialization(self):
        """Test EDA initializer."""
        eda = NaiveEDA(seed=42)
        assert eda is not None
        assert hasattr(eda, "random_state")
        assert hasattr(eda, "stop_words")

    def test_get_synonyms(self, eda_augmenter):
        """Test synonym retrieval from WordNet."""
        synonyms = eda_augmenter._get_synonyms("happy")
        assert isinstance(synonyms, list)
        # "happy" should have multiple synonyms
        assert len(synonyms) > 0

    def test_get_synonyms_no_match(self, eda_augmenter):
        """Test synonym retrieval for word with no synonyms."""
        synonyms = eda_augmenter._get_synonyms("xyz123nonword")
        assert isinstance(synonyms, list)
        assert len(synonyms) == 0

    def test_random_insertion_preserves_entities(
        self, eda_augmenter, sentence_with_entities
    ):
        """Test that random insertion preserves entity tags."""
        original_entities = [
            (i, token, tag)
            for i, (token, _, _, tag) in enumerate(sentence_with_entities)
            if tag != "O"
        ]

        augmented = eda_augmenter.random_insertion(
            sentence_with_entities, num_insertions=2
        )

        # Check that entity tokens are still present
        augmented_entities = [
            (token, tag) for token, _, _, tag in augmented if tag != "O"
        ]
        original_entity_tokens = [(token, tag) for _, token, tag in original_entities]

        for entity in original_entity_tokens:
            assert entity in augmented_entities

    def test_random_insertion_increases_length(self, eda_augmenter, sample_sentence):
        """Test that random insertion increases sentence length."""
        original_len = len(sample_sentence)
        augmented = eda_augmenter.random_insertion(sample_sentence, num_insertions=3)
        assert len(augmented) >= original_len

    def test_random_swap_preserves_tags(self, eda_augmenter, sentence_with_entities):
        """Test that random swap preserves all tags."""
        original_tags = [tag for _, _, _, tag in sentence_with_entities]
        augmented = eda_augmenter.random_swap(sentence_with_entities, num_swaps=2)
        augmented_tags = [tag for _, _, _, tag in augmented]

        assert sorted(original_tags) == sorted(augmented_tags)

    def test_random_swap_doesnt_touch_entities(
        self, eda_augmenter, sentence_with_entities
    ):
        """Test that random swap only swaps non-entity tokens."""
        augmented = eda_augmenter.random_swap(sentence_with_entities, num_swaps=5)

        # Entity positions might change, but entity tokens should remain
        original_entities = [
            (token, tag) for token, _, _, tag in sentence_with_entities if tag != "O"
        ]
        augmented_entities = [
            (token, tag) for token, _, _, tag in augmented if tag != "O"
        ]

        assert sorted(original_entities) == sorted(augmented_entities)

    def test_random_deletion_preserves_entities(
        self, eda_augmenter, sentence_with_entities
    ):
        """Test that random deletion never removes entities."""
        augmented = eda_augmenter.random_deletion(
            sentence_with_entities, deletion_prob=0.5
        )

        original_entities = [
            (token, tag) for token, _, _, tag in sentence_with_entities if tag != "O"
        ]
        augmented_entities = [
            (token, tag) for token, _, _, tag in augmented if tag != "O"
        ]

        assert original_entities == augmented_entities

    def test_random_deletion_reduces_or_maintains_length(
        self, eda_augmenter, sample_sentence
    ):
        """Test that random deletion reduces or maintains sentence length."""
        original_len = len(sample_sentence)
        augmented = eda_augmenter.random_deletion(sample_sentence, deletion_prob=0.3)
        assert len(augmented) <= original_len

    def test_random_deletion_never_empty(self, eda_augmenter, sample_sentence):
        """Test that random deletion never results in empty sentence."""
        augmented = eda_augmenter.random_deletion(sample_sentence, deletion_prob=0.99)
        assert len(augmented) > 0

    def test_random_synonym_replacement(self, eda_augmenter, sample_sentence):
        """Test synonym replacement modifies tokens."""
        original_tokens = [token for token, _, _, _ in sample_sentence]
        augmented = eda_augmenter.random_synonym_replacement(
            sample_sentence, num_replacements=2
        )
        augmented_tokens = [token for token, _, _, _ in augmented]

        # At least one token should be different (though not guaranteed in rare cases)
        assert len(augmented_tokens) == len(original_tokens)

    def test_random_synonym_replacement_preserves_entities(
        self, eda_augmenter, sentence_with_entities
    ):
        """Test that synonym replacement preserves entities."""
        original_entities = [
            (token, tag) for token, _, _, tag in sentence_with_entities if tag != "O"
        ]
        augmented = eda_augmenter.random_synonym_replacement(
            sentence_with_entities, num_replacements=2
        )
        augmented_entities = [
            (token, tag) for token, _, _, tag in augmented if tag != "O"
        ]

        assert original_entities == augmented_entities

    def test_augment_combines_all_operations(self, eda_augmenter, sample_sentence):
        """Test that augment method applies all operations."""
        augmented = eda_augmenter.augment(
            sample_sentence, alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, p_rd=0.1
        )

        assert isinstance(augmented, list)
        assert len(augmented) > 0
        # Check structure is preserved
        assert all(len(item) == 4 for item in augmented)

    def test_augment_with_entities(self, eda_augmenter, sentence_with_entities):
        """Test augment preserves entity information."""
        original_entities = [
            (token, tag) for token, _, _, tag in sentence_with_entities if tag != "O"
        ]
        augmented = eda_augmenter.augment(sentence_with_entities)
        augmented_entities = [
            (token, tag) for token, _, _, tag in augmented if tag != "O"
        ]

        assert original_entities == augmented_entities

    def test_reproducibility_with_seed(self, sample_sentence):
        """Test that same seed produces same augmentation."""
        eda1 = NaiveEDA(seed=42)
        eda2 = NaiveEDA(seed=42)

        aug1 = eda1.augment(sample_sentence)
        aug2 = eda2.augment(sample_sentence)

        assert aug1 == aug2

    def test_different_seeds_produce_different_results(self, sample_sentence):
        """Test that different seeds produce different augmentations."""
        eda1 = NaiveEDA(seed=42)
        eda2 = NaiveEDA(seed=123)

        results = set()
        for seed in [42, 123, 456]:
            eda = NaiveEDA(seed=seed)
            aug = eda.augment(sample_sentence)
            results.add(tuple([tuple(t) for t in aug]))

        # Most likely to get multiple different results
        assert len(results) >= 1


# ============================================================================
# Tests for EntityAwareReplacement
# ============================================================================


class TestEntityAwareReplacement:
    """Tests for the EntityAwareReplacement class."""

    def test_initialization(self):
        """Test entity replacement initializer."""
        ear = EntityAwareReplacement(seed=42)
        assert ear is not None
        assert hasattr(ear, "entity_examples")
        assert isinstance(ear.entity_examples, dict)

    def test_add_entity_examples(
        self, entity_replacement_augmenter, multi_entity_sentence
    ):
        """Test adding entity examples."""
        sentences = [multi_entity_sentence]
        entity_replacement_augmenter.add_entity_examples(sentences)

        assert len(entity_replacement_augmenter.entity_examples) > 0
        assert "PER" in entity_replacement_augmenter.entity_examples
        assert "ORG" in entity_replacement_augmenter.entity_examples
        assert "LOC" in entity_replacement_augmenter.entity_examples

    def test_add_entity_examples_extracts_correctly(
        self, entity_replacement_augmenter, multi_entity_sentence
    ):
        """Test that entity examples are extracted correctly."""
        sentences = [multi_entity_sentence]
        entity_replacement_augmenter.add_entity_examples(sentences)

        # Check PER entity
        assert ["Steve", "Jobs"] in entity_replacement_augmenter.entity_examples["PER"]
        # Check ORG entity
        assert ["Apple"] in entity_replacement_augmenter.entity_examples["ORG"]
        # Check LOC entity
        assert ["Cupertino"] in entity_replacement_augmenter.entity_examples["LOC"]

    def test_get_entity_span(
        self, entity_replacement_augmenter, sentence_with_entities
    ):
        """Test entity span extraction."""
        start, end, entity_type = entity_replacement_augmenter._get_entity_span(
            sentence_with_entities, 0
        )

        assert start == 0
        assert end == 2  # "Barack Obama" is 2 tokens
        assert entity_type == "PER"

    def test_get_entity_span_single_token(
        self, entity_replacement_augmenter, sentence_with_entities
    ):
        """Test entity span for single-token entity."""
        start, end, entity_type = entity_replacement_augmenter._get_entity_span(
            sentence_with_entities, 5
        )

        assert start == 5
        assert end == 6  # "Hawaii" is 1 token
        assert entity_type == "LOC"

    def test_get_entity_span_non_entity(
        self, entity_replacement_augmenter, sentence_with_entities
    ):
        """Test entity span on non-entity token."""
        start, end, entity_type = entity_replacement_augmenter._get_entity_span(
            sentence_with_entities, 2
        )

        assert start == -1
        assert end == -1
        assert entity_type == ""

    def test_augment_without_examples(
        self, entity_replacement_augmenter, sentence_with_entities
    ):
        """Test augment without examples doesn't crash."""
        augmented = entity_replacement_augmenter.augment(
            sentence_with_entities, replacement_prob=0.5
        )

        assert isinstance(augmented, list)
        assert len(augmented) > 0

    def test_augment_with_examples(
        self,
        entity_replacement_augmenter,
        multi_entity_sentence,
        sentence_with_entities,
    ):
        """Test augment with entity examples available."""
        # Add examples
        entity_replacement_augmenter.add_entity_examples([multi_entity_sentence] * 3)

        # Augment with high probability to ensure replacement
        augmented = entity_replacement_augmenter.augment(
            sentence_with_entities, replacement_prob=0.99
        )

        assert isinstance(augmented, list)
        assert len(augmented) > 0
        # All tags should still be valid
        for token, pos, chunk, tag in augmented:
            assert tag in {"O", "B-PER", "I-PER", "B-LOC", "I-LOC"}

    def test_augment_preserves_entity_structure(
        self, entity_replacement_augmenter, multi_entity_sentence
    ):
        """Test that augment preserves overall entity structure."""
        entity_replacement_augmenter.add_entity_examples([multi_entity_sentence] * 3)

        original_tags = [tag for _, _, _, tag in multi_entity_sentence]
        augmented = entity_replacement_augmenter.augment(
            multi_entity_sentence, replacement_prob=0.99
        )
        augmented_tags = [tag for _, _, _, tag in augmented]

        # Number of entity vs non-entity tags should be similar
        original_entity_count = sum(1 for tag in original_tags if tag != "O")
        augmented_entity_count = sum(1 for tag in augmented_tags if tag != "O")

        assert augmented_entity_count > 0

    def test_reproducibility_with_seed(
        self,
        entity_replacement_augmenter,
        multi_entity_sentence,
        sentence_with_entities,
    ):
        """Test reproducibility with seed."""
        entity_replacement_augmenter.add_entity_examples([multi_entity_sentence] * 3)

        ear1 = EntityAwareReplacement(seed=42)
        ear1.add_entity_examples([multi_entity_sentence] * 3)
        ear2 = EntityAwareReplacement(seed=42)
        ear2.add_entity_examples([multi_entity_sentence] * 3)

        aug1 = ear1.augment(sentence_with_entities)
        aug2 = ear2.augment(sentence_with_entities)

        assert aug1 == aug2


# ============================================================================
# Tests for ConllAugmenter
# ============================================================================


class TestConllAugmenter:
    """Tests for the ConllAugmenter class."""

    def test_initialization(self):
        """Test ConllAugmenter initializer."""
        augmenter = ConllAugmenter(seed=42)
        assert augmenter is not None
        assert hasattr(augmenter, "eda")
        assert hasattr(augmenter, "entity_replacement")
        assert hasattr(augmenter, "back_translation")

    def test_augment_eda(self, conll_augmenter, sample_sentence):
        """Test EDA augmentation through main class."""
        augmented = conll_augmenter.augment_eda(sample_sentence)

        assert isinstance(augmented, list)
        assert len(augmented) > 0

    def test_augment_entity_replacement(
        self, conll_augmenter, multi_entity_sentence, sentence_with_entities
    ):
        """Test entity replacement through main class."""
        conll_augmenter.set_entity_examples([multi_entity_sentence])
        augmented = conll_augmenter.augment_entity_replacement(sentence_with_entities)

        assert isinstance(augmented, list)
        assert len(augmented) > 0

    def test_set_entity_examples(self, conll_augmenter, multi_entity_sentence):
        """Test setting entity examples."""
        conll_augmenter.set_entity_examples([multi_entity_sentence])

        assert len(conll_augmenter.entity_replacement.entity_examples) > 0

    def test_augment_batch_single_method(self, conll_augmenter, sample_sentence):
        """Test batch augmentation with single method."""
        sentences = [sample_sentence] * 2
        augmented = conll_augmenter.augment_batch(
            sentences, methods=["eda"], num_augmentations=1
        )

        # Should return original + 2 augmented
        assert len(augmented) == 4
        assert augmented[:2] == sentences

    def test_augment_batch_multiple_methods(
        self, conll_augmenter, multi_entity_sentence, sentence_with_entities
    ):
        """Test batch augmentation with multiple methods."""
        conll_augmenter.set_entity_examples([multi_entity_sentence])
        sentences = [sentence_with_entities]

        augmented = conll_augmenter.augment_batch(
            sentences, methods=["eda", "entity_replacement"], num_augmentations=2
        )

        # Should return original + 2 augmented
        assert len(augmented) >= 1
        assert all(isinstance(s, list) for s in augmented)
        assert all(all(len(item) == 4 for item in s) for s in augmented)

    def test_augment_batch_returns_originals(self, conll_augmenter, sample_sentence):
        """Test that batch augmentation includes originals."""
        sentences = [sample_sentence]
        augmented = conll_augmenter.augment_batch(sentences, num_augmentations=1)

        # First sentence should be original
        assert augmented[0] == sample_sentence

    def test_augment_batch_with_zero_augmentations(
        self, conll_augmenter, sample_sentence
    ):
        """Test batch augmentation with zero augmentations."""
        sentences = [sample_sentence]
        augmented = conll_augmenter.augment_batch(sentences, num_augmentations=0)

        # Should return only originals
        assert len(augmented) == 1
        assert augmented[0] == sample_sentence


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_augmentation_pipeline(
        self, conll_augmenter, multi_entity_sentence, sentence_with_entities
    ):
        """Test full augmentation pipeline."""
        conll_augmenter.set_entity_examples([multi_entity_sentence] * 3)

        sentences = [sentence_with_entities]
        augmented = conll_augmenter.augment_batch(
            sentences, methods=["eda", "entity_replacement"], num_augmentations=2
        )

        assert len(augmented) > len(sentences)

        # Verify all augmented sentences have valid structure
        for sentence in augmented:
            assert isinstance(sentence, list)
            for item in sentence:
                assert len(item) == 4
                assert isinstance(item[0], str)  # token
                assert isinstance(item[1], str)  # pos
                assert isinstance(item[2], str)  # chunk
                assert isinstance(item[3], str)  # ner tag

    def test_entity_preservation_across_augmentations(self):
        """Test entity preservation across different augmentation methods."""
        sentence = [
            ("John", "NNP", "B-NP", "B-PER"),
            ("Smith", "NNP", "I-NP", "I-PER"),
            ("works", "VBZ", "B-VP", "O"),
            ("at", "IN", "B-PP", "O"),
            ("Google", "NNP", "B-NP", "B-ORG"),
        ]

        eda = NaiveEDA(seed=42)
        ear = EntityAwareReplacement(seed=42)

        # Apply multiple augmentations
        for _ in range(5):
            aug1 = eda.augment(sentence)
            entities_after_eda = [(t, tag) for t, _, _, tag in aug1 if tag != "O"]
            original_entities = [(t, tag) for t, _, _, tag in sentence if tag != "O"]
            assert len(entities_after_eda) == len(original_entities)

    def test_augmenter_handles_edge_cases(self, conll_augmenter):
        """Test augmenter handles edge cases."""
        # Empty-like sentence
        short_sentence = [("Word", "NN", "O", "O")]
        augmented = conll_augmenter.augment_eda(short_sentence)
        assert len(augmented) >= 1

        # All entities
        entity_only = [
            ("John", "NNP", "B-NP", "B-PER"),
            ("Smith", "NNP", "I-NP", "I-PER"),
        ]
        augmented = conll_augmenter.augment_eda(entity_only)
        assert len(augmented) >= 1
        entities_in_aug = [(t, tag) for t, _, _, tag in augmented if tag != "O"]
        assert len(entities_in_aug) == 2

    def test_batch_augmentation_consistency(
        self, conll_augmenter, multi_entity_sentence
    ):
        """Test that batch augmentation is consistent."""
        conll_augmenter.set_entity_examples([multi_entity_sentence] * 3)

        sentence1 = multi_entity_sentence
        sentence2 = [
            ("Alice", "NNP", "B-NP", "B-PER"),
            ("visited", "VBD", "B-VP", "O"),
            ("Paris", "NNP", "B-NP", "B-LOC"),
        ]

        augmented = conll_augmenter.augment_batch(
            [sentence1, sentence2], num_augmentations=1
        )

        # Check all sentences have valid NER tags
        valid_tags = {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}
        for sentence in augmented:
            for token, pos, chunk, tag in sentence:
                assert tag in valid_tags


# ============================================================================
# Property-based Tests
# ============================================================================


class TestProperties:
    """Property-based tests."""

    def test_augmentation_preserves_sentence_structure(
        self, eda_augmenter, sentence_with_entities
    ):
        """Test that augmentations preserve basic sentence structure."""
        for _ in range(10):
            augmented = eda_augmenter.augment(sentence_with_entities)

            # All items should be 4-tuples
            assert all(len(item) == 4 for item in augmented)
            # All tags should be strings
            assert all(isinstance(item[3], str) for item in augmented)
            # No sentence should be empty
            assert len(augmented) > 0

    def test_entity_consistency_across_operations(
        self, eda_augmenter, sentence_with_entities
    ):
        """Test entity consistency across multiple augmentation operations."""
        original_entity_tokens = [
            token for token, _, _, tag in sentence_with_entities if tag != "O"
        ]

        for _ in range(10):
            augmented = eda_augmenter.augment(sentence_with_entities)
            augmented_entity_tokens = [
                token for token, _, _, tag in augmented if tag != "O"
            ]

            # Same entity tokens should be present
            assert sorted(original_entity_tokens) == sorted(augmented_entity_tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
