"""
Comprehensive test suite for augmentation.py using pytest.
Ensures that augmentation operations strictly adhere to boundary conditions 
(e.g., not modifying entity tokens during EDA) and proper label alignment.
"""

import sys
from pathlib import Path
import pytest
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from augmentation import (
    augment_naive_eda,
    augment_contextual_mlm,
    augment_back_translation,
    augment_entity_aware
)

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_sentence():
    tokens = ["The", "CEO", "of", "Microsoft", "is", "Satya", "Nadella", "today"]
    labels = ["O", "O", "O", "B-ORG", "O", "B-PER", "I-PER", "O"]
    return tokens, labels

@pytest.fixture
def mock_synonym_dict():
    return {
        "the": ["a", "an"],
        "ceo": ["chief", "director"],
        "today": ["now", "currently"]
    }

@pytest.fixture
def mock_entity_kb():
    return {
        "PER": [["Bill", "Gates"], ["John"]],
        "ORG": [["Apple", "Inc."], ["Google"]]
    }

class TestNaiveEDA:
    def test_eda_preserves_entities(self, sample_sentence, mock_synonym_dict):
        """Ensure Naive EDA never alters tokens labeled with B-* or I-*"""
        tokens, labels = sample_sentence
        # Using alpha=1.0 guarantees all 'O' tokens get perturbed
        aug_tokens, aug_labels = augment_naive_eda(tokens, labels, mock_synonym_dict, alpha=1.0)
        
        # Verify the entity tokens survived the augmentation
        assert "Microsoft" in aug_tokens, "Microsoft should not be deleted"
        assert "Satya" in aug_tokens, "Satya should not be deleted"
        assert "Nadella" in aug_tokens, "Nadella should not be deleted"
        
        # Verify their labels remain correctly aligned despite any shifting
        ms_idx = aug_tokens.index("Microsoft")
        assert aug_labels[ms_idx] == "B-ORG"
        
        satya_idx = aug_tokens.index("Satya")
        assert aug_labels[satya_idx] == "B-PER"
        
        nadella_idx = aug_tokens.index("Nadella")
        assert aug_labels[nadella_idx] == "I-PER"
        
    def test_eda_deletion_reduces_length(self, sample_sentence, mock_synonym_dict, monkeypatch):
        """Force a deletion operation and verify sequence length is reduced appropriately."""
        tokens, labels = sample_sentence
        
        # Force random choice to always pick 'delete'
        monkeypatch.setattr("random.choice", lambda x: "delete" if isinstance(x, list) and "delete" in x else x[0])
        
        aug_tokens, aug_labels = augment_naive_eda(tokens, labels, mock_synonym_dict, alpha=0.5)
        
        assert len(aug_tokens) < len(tokens)
        assert len(aug_tokens) == len(aug_labels)

class TestEntityAwareReplacement:
    def test_entity_replacement_label_alignment(self, sample_sentence, mock_entity_kb, monkeypatch):
        """Ensure replacing an entity maps the BIO tags to the exact length of the new span."""
        tokens, labels = sample_sentence
        
        # Force the choice to replace 'Microsoft' with 'Apple Inc.'
        monkeypatch.setattr("random.choice", lambda x: (3, 3, "ORG") if isinstance(x, list) and len(x) > 0 and isinstance(x[0], tuple) else ["Apple", "Inc."])
        
        aug_tokens, aug_labels = augment_entity_aware(tokens, labels, mock_entity_kb)
        
        # Original: "The CEO of Microsoft is..." (length 8)
        # New: "The CEO of Apple Inc. is..." (length 9)
        assert len(aug_tokens) == 9
        assert len(aug_labels) == 9
        
        assert aug_tokens[3] == "Apple"
        assert aug_tokens[4] == "Inc."
        assert aug_labels[3] == "B-ORG"
        assert aug_labels[4] == "I-ORG"

class TestModelPlaceholders:
    def test_mlm_requires_model(self, sample_sentence):
        tokens, labels = sample_sentence
        with pytest.raises(ValueError, match="mlm_pipeline cannot be None"):
            augment_contextual_mlm(tokens, labels, mlm_pipeline=None)

    def test_backtranslation_requires_models(self, sample_sentence):
        tokens, labels = sample_sentence
        with pytest.raises(ValueError, match="Translation models and alignment heuristic must be provided"):
            augment_back_translation(tokens, labels, None, None, None)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])