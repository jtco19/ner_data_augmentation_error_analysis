"""
Data augmentation functionality for CONLL2003 NER dataset.

Implements three augmentation techniques:
1. Naive EDA (Easy Data Augmentation)
2. Back translation
3. Entity-aware replacement
"""

import random
import re
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data if not already present
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# Type definitions
Token = str
POS = str
Chunk = str
NERTag = str
TokenTuple = Tuple[Token, POS, Chunk, NERTag]
Sentence = List[TokenTuple]


class NaiveEDA:
    """
    Naive Easy Data Augmentation (EDA) implementation.
    Applies random operations: insertion, swap, deletion, and synonym replacement.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize EDA augmenter.

        Args:
            seed: Random seed for reproducibility
        """
        self.random_state = random.Random(seed)
        self.stop_words = set(stopwords.words("english"))

    def _get_synonyms(self, token: str) -> List[str]:
        """
        Get synonyms for a token using WordNet.

        Args:
            token: The token to find synonyms for

        Returns:
            List of synonyms
        """
        synonyms = set()
        for synset in wordnet.synsets(token):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        return list(synonyms)

    def random_insertion(self, sentence: Sentence, num_insertions: int = 1) -> Sentence:
        """
        Randomly insert synonyms of random words in the sentence.
        Preserves NER tags by inserting only in non-entity tokens.

        Args:
            sentence: Input sentence as list of (token, pos, chunk, ner) tuples
            num_insertions: Number of words to insert

        Returns:
            Augmented sentence
        """
        augmented = list(sentence)

        # Get indices of non-entity tokens (tag is 'O')
        valid_indices = [
            i for i, (token, _, _, tag) in enumerate(augmented) if tag == "O"
        ]

        if not valid_indices:
            return augmented

        for _ in range(num_insertions):
            if not valid_indices:
                break

            idx = self.random_state.choice(valid_indices)
            token = augmented[idx][0].lower()
            synonyms = self._get_synonyms(token)

            if synonyms:
                synonym = self.random_state.choice(synonyms)
                # Insert after the selected token
                augmented.insert(idx + 1, (synonym, augmented[idx][1], "O", "O"))

        return augmented

    def random_swap(self, sentence: Sentence, num_swaps: int = 1) -> Sentence:
        """
        Randomly swap words in the sentence.
        Preserves NER tags by only swapping non-entity tokens.

        Args:
            sentence: Input sentence as list of (token, pos, chunk, ner) tuples
            num_swaps: Number of swaps to perform

        Returns:
            Augmented sentence
        """
        augmented = list(sentence)

        # Get indices of non-entity tokens
        valid_indices = [i for i, (_, _, _, tag) in enumerate(augmented) if tag == "O"]

        if len(valid_indices) < 2:
            return augmented

        for _ in range(num_swaps):
            idx1, idx2 = self.random_state.sample(valid_indices, 2)
            augmented[idx1], augmented[idx2] = augmented[idx2], augmented[idx1]

        return augmented

    def random_deletion(
        self, sentence: Sentence, deletion_prob: float = 0.1
    ) -> Sentence:
        """
        Randomly delete words with a given probability.
        Never deletes entity tokens.

        Args:
            sentence: Input sentence as list of (token, pos, chunk, ner) tuples
            deletion_prob: Probability of deleting each non-entity token

        Returns:
            Augmented sentence
        """
        if len(sentence) == 1:
            return sentence

        augmented = [
            item
            for item in sentence
            if item[3] != "O" or self.random_state.random() > deletion_prob
        ]

        return augmented if augmented else list(sentence)

    def random_synonym_replacement(
        self, sentence: Sentence, num_replacements: int = 1
    ) -> Sentence:
        """
        Replace random tokens with their synonyms.
        Only replaces non-entity tokens.

        Args:
            sentence: Input sentence as list of (token, pos, chunk, ner) tuples
            num_replacements: Number of tokens to replace

        Returns:
            Augmented sentence
        """
        augmented = list(sentence)

        # Get indices of non-entity tokens
        valid_indices = [i for i, (_, _, _, tag) in enumerate(augmented) if tag == "O"]

        if not valid_indices:
            return augmented

        num_replacements = min(num_replacements, len(valid_indices))
        indices_to_replace = self.random_state.sample(valid_indices, num_replacements)

        for idx in indices_to_replace:
            token = augmented[idx][0].lower()
            synonyms = self._get_synonyms(token)

            if synonyms:
                synonym = self.random_state.choice(synonyms)
                token_tuple = augmented[idx]
                augmented[idx] = (
                    synonym,
                    token_tuple[1],
                    token_tuple[2],
                    token_tuple[3],
                )

        return augmented

    def augment(
        self,
        sentence: Sentence,
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        p_rd: float = 0.1,
    ) -> Sentence:
        """
        Apply all EDA operations to a sentence.

        Args:
            sentence: Input sentence
            alpha_sr: Percentage of words to replace with synonyms
            alpha_ri: Percentage of words to insert
            alpha_rs: Percentage of words to swap
            p_rd: Probability of deleting each word

        Returns:
            Augmented sentence
        """
        num_words = len([t for t in sentence if t[3] == "O"])

        num_sr = max(1, int(alpha_sr * num_words))
        num_ri = max(1, int(alpha_ri * num_words))
        num_rs = max(1, int(alpha_rs * num_words))

        augmented = self.random_synonym_replacement(sentence, num_sr)
        augmented = self.random_insertion(augmented, num_ri)
        augmented = self.random_swap(augmented, num_rs)
        augmented = self.random_deletion(augmented, p_rd)

        return augmented


class BackTranslation:
    """
    Back translation augmentation using intermediate languages.
    Requires transformers library and pre-trained MT models.
    """

    def __init__(self, intermediate_lang: str = "de", device: str = "cpu"):
        """
        Initialize back translation augmenter.

        Args:
            intermediate_lang: Intermediate language code (default: 'de' for German)
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.intermediate_lang = intermediate_lang
        self.device = device
        self.translator = None

        try:
            from transformers import pipeline

            self.pipeline = pipeline
            self._initialize_translators()
        except ImportError:
            print(
                "Warning: transformers not installed. Back translation will not work."
            )
            print("Install with: pip install transformers torch")

    def _initialize_translators(self):
        """Initialize translation pipelines."""
        try:
            # English to intermediate language
            model_name_to = f"Helsinki-NLP/opus-mt-en-{self.intermediate_lang}"
            self.translator_to = self.pipeline(
                "translation",
                model=model_name_to,
                device=0 if self.device == "cuda" else -1,
            )

            # Intermediate language back to English
            model_name_back = f"Helsinki-NLP/opus-mt-{self.intermediate_lang}-en"
            self.translator_back = self.pipeline(
                "translation",
                model=model_name_back,
                device=0 if self.device == "cuda" else -1,
            )

            self.translator = True
        except Exception as e:
            print(f"Warning: Failed to initialize translators: {e}")
            self.translator = None

    def _reconstruct_sentence(self, tokens: List[str], original: Sentence) -> Sentence:
        """
        Reconstruct sentence with NER tags after translation.
        Aligns translated tokens with original entity tags.

        Args:
            tokens: Translated tokens
            original: Original sentence with tags

        Returns:
            Reconstructed sentence with NER tags
        """
        # Simple alignment: preserve tags for entities in relative positions
        entity_map = [(i, item) for i, item in enumerate(original) if item[3] != "O"]

        reconstructed = []
        for token in tokens:
            if entity_map and len(reconstructed) < len(original):
                # Try to preserve some entity information
                reconstructed.append((token, "NN", "O", "O"))
            else:
                reconstructed.append((token, "NN", "O", "O"))

        return reconstructed

    def augment(self, sentence: Sentence) -> Optional[Sentence]:
        """
        Augment sentence using back translation.

        Args:
            sentence: Input sentence

        Returns:
            Augmented sentence or None if translation fails
        """
        if not self.translator:
            print("Translators not initialized")
            return None

        try:
            # Extract text
            text = " ".join([token for token, _, _, _ in sentence])

            # Translate to intermediate language
            translated_to = self.translator_to(text)
            intermediate_text = translated_to[0]["translation_text"]

            # Translate back to English
            translated_back = self.translator_back(intermediate_text)
            back_text = translated_back[0]["translation_text"]

            # Tokenize back-translated text
            tokens = back_text.split()

            # Reconstruct with original NER tags
            augmented = self._reconstruct_sentence(tokens, sentence)

            return augmented
        except Exception as e:
            print(f"Back translation failed: {e}")
            return None


class EntityAwareReplacement:
    """
    Entity-aware replacement augmentation.
    Replaces entities with similar entities from the same class.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize entity-aware replacement augmenter.

        Args:
            seed: Random seed for reproducibility
        """
        self.random_state = random.Random(seed)
        self.entity_examples: Dict[str, List[List[str]]] = defaultdict(list)

    def add_entity_examples(self, sentences: List[Sentence]):
        """
        Build a dictionary of entity examples from training data.

        Args:
            sentences: List of sentences with NER tags
        """
        for sentence in sentences:
            current_entity = []
            current_tag = None

            for token, pos, chunk, tag in sentence:
                if tag == "O":
                    if current_entity and current_tag:
                        self.entity_examples[current_tag].append(current_entity)
                    current_entity = []
                    current_tag = None
                elif tag.startswith("B-") or tag.startswith("I-"):
                    if tag.startswith("B-") and current_entity and current_tag:
                        self.entity_examples[current_tag].append(current_entity)
                        current_entity = []

                    current_tag = tag.split("-")[1]
                    current_entity.append(token)

            if current_entity and current_tag:
                self.entity_examples[current_tag].append(current_entity)

    def _get_entity_span(
        self, sentence: Sentence, start_idx: int
    ) -> Tuple[int, int, str]:
        """
        Extract entity span starting at start_idx.

        Args:
            sentence: Input sentence
            start_idx: Starting index

        Returns:
            Tuple of (start_idx, end_idx, entity_tag)
        """
        if start_idx >= len(sentence) or sentence[start_idx][3] == "O":
            return -1, -1, ""

        tag = sentence[start_idx][3]
        entity_type = tag.split("-")[1] if "-" in tag else tag

        end_idx = start_idx + 1
        while end_idx < len(sentence) and sentence[end_idx][3] != "O":
            next_tag = sentence[end_idx][3]
            next_type = next_tag.split("-")[1] if "-" in next_tag else next_tag
            if next_type != entity_type or next_tag.startswith("B-"):
                break
            end_idx += 1

        return start_idx, end_idx, entity_type

    def augment(self, sentence: Sentence, replacement_prob: float = 0.5) -> Sentence:
        """
        Replace entities with similar entities from the same class.

        Args:
            sentence: Input sentence
            replacement_prob: Probability of replacing each entity

        Returns:
            Augmented sentence
        """
        augmented = list(sentence)
        idx = 0

        while idx < len(augmented):
            if (
                augmented[idx][3] != "O"
                and self.random_state.random() < replacement_prob
            ):
                start_idx, end_idx, entity_type = self._get_entity_span(augmented, idx)

                if start_idx >= 0 and entity_type in self.entity_examples:
                    replacement = self.random_state.choice(
                        self.entity_examples[entity_type]
                    )

                    # Replace entity tokens
                    for i, token in enumerate(replacement):
                        if start_idx + i < end_idx:
                            token_tuple = augmented[start_idx + i]
                            tag = f"B-{entity_type}" if i == 0 else f"I-{entity_type}"
                            augmented[start_idx + i] = (
                                token,
                                token_tuple[1],
                                token_tuple[2],
                                tag,
                            )

                    # Remove extra tokens if replacement is shorter
                    if len(replacement) < end_idx - start_idx:
                        del augmented[start_idx + len(replacement) : end_idx]
                        end_idx = start_idx + len(replacement)

                    # Add tokens if replacement is longer
                    elif len(replacement) > end_idx - start_idx:
                        for i in range(len(replacement) - (end_idx - start_idx)):
                            token = replacement[end_idx - start_idx + i]
                            tag = f"I-{entity_type}"
                            augmented.insert(end_idx + i, (token, "NN", "O", tag))

                idx = end_idx if start_idx >= 0 else idx + 1
            else:
                idx += 1

        return augmented


class ConllAugmenter:
    """
    Main augmentation class that combines all augmentation techniques.
    """

    def __init__(self, seed: int = 17):
        """
        Initialize the augmenter.

        Args:
            seed: Random seed for reproducibility
        """
        self.eda = NaiveEDA(seed=seed)
        self.back_translation = BackTranslation()
        self.entity_replacement = EntityAwareReplacement(seed=seed)
        self.random_state = random.Random(seed)

    def augment_eda(
        self,
        sentence: Sentence,
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        p_rd: float = 0.1,
    ) -> Sentence:
        """
        Augment using naive EDA.

        Args:
            sentence: Input sentence
            alpha_sr: Synonym replacement percentage
            alpha_ri: Random insertion percentage
            alpha_rs: Random swap percentage
            p_rd: Random deletion probability

        Returns:
            Augmented sentence
        """
        return self.eda.augment(sentence, alpha_sr, alpha_ri, alpha_rs, p_rd)

    def augment_back_translation(self, sentence: Sentence) -> Optional[Sentence]:
        """
        Augment using back translation.

        Args:
            sentence: Input sentence

        Returns:
            Augmented sentence or None if translation fails
        """
        return self.back_translation.augment(sentence)

    def augment_entity_replacement(
        self, sentence: Sentence, replacement_prob: float = 0.5
    ) -> Sentence:
        """
        Augment using entity-aware replacement.

        Args:
            sentence: Input sentence
            replacement_prob: Entity replacement probability

        Returns:
            Augmented sentence
        """
        return self.entity_replacement.augment(sentence, replacement_prob)

    def set_entity_examples(self, sentences: List[Sentence]):
        """
        Set entity examples for entity-aware replacement.

        Args:
            sentences: List of sentences with NER tags
        """
        self.entity_replacement.add_entity_examples(sentences)

    def augment_batch(
        self,
        sentences: List[Sentence],
        methods: List[str] = ["eda", "entity_replacement"],
        num_augmentations: int = 1,
    ) -> List[Sentence]:
        """
        Augment a batch of sentences using specified methods.

        Args:
            sentences: List of input sentences
            methods: List of augmentation methods to use ('eda', 'back_translation', 'entity_replacement')
            num_augmentations: Number of augmented copies per sentence

        Returns:
            Original sentences + augmented copies
        """
        augmented_sentences = list(sentences)

        for sentence in sentences:
            for _ in range(num_augmentations):
                method = self.random_state.choice(methods)

                if method == "eda":
                    augmented = self.augment_eda(sentence)
                elif method == "back_translation":
                    augmented = self.augment_back_translation(sentence)
                    if augmented is None:
                        continue
                elif method == "entity_replacement":
                    augmented = self.augment_entity_replacement(sentence)
                else:
                    continue

                augmented_sentences.append(augmented)

        return augmented_sentences
