"""
Data augmentation module for Named Entity Recognition.
Implements Naïve EDA (non-entity restricted), Contextual MLM, Back Translation, 
and Entity-Aware Replacement with comprehensive logging.
"""

import logging
import random
import copy
from typing import List, Tuple, Dict, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def augment_naive_eda(
    tokens: List[str], 
    labels: List[str], 
    synonym_dict: Dict[str, List[str]], 
    alpha: float = 0.1
) -> Tuple[List[str], List[str]]:
    """
    Naïve EDA (Synonym Replacement, Swap, Deletion) restricted strictly to non-entity tokens.
    
    Args:
        tokens: Original token sequence.
        labels: Original BIO label sequence.
        synonym_dict: A dictionary mapping words to their synonyms. YOU MUST PROVIDE THIS DATA.
        alpha: Probability of perturbing a given non-entity token.
        
    Returns:
        Tuple of (augmented_tokens, augmented_labels)
    """
    aug_tokens = copy.deepcopy(tokens)
    aug_labels = copy.deepcopy(labels)
    
    # Identify non-entity indices
    o_indices = [i for i, label in enumerate(labels) if label == "O"]
    
    if not o_indices:
        logger.debug("No 'O' tokens available for EDA perturbation.")
        return aug_tokens, aug_labels

    num_perturbations = max(1, int(alpha * len(o_indices)))
    target_indices = random.sample(o_indices, min(num_perturbations, len(o_indices)))

    for idx in target_indices:
        operation = random.choice(["synonym", "swap", "delete"])
        
        if operation == "synonym":
            word = aug_tokens[idx].lower()
            if word in synonym_dict and synonym_dict[word]:
                aug_tokens[idx] = random.choice(synonym_dict[word])
                logger.debug(f"EDA Synonym: replaced {word} with {aug_tokens[idx]}")
                
        elif operation == "swap":
            # Swap with another adjacent 'O' token if possible
            if idx > 0 and aug_labels[idx-1] == "O":
                aug_tokens[idx], aug_tokens[idx-1] = aug_tokens[idx-1], aug_tokens[idx]
                logger.debug(f"EDA Swap: swapped indices {idx} and {idx-1}")
                
        elif operation == "delete":
            # Mark for deletion
            aug_tokens[idx] = "[DEL]"
            aug_labels[idx] = "[DEL]"

    # Filter out deletions
    final_tokens = [t for t in aug_tokens if t != "[DEL]"]
    final_labels = [l for l in aug_labels if l != "[DEL]"]
    
    return final_tokens, final_labels


def augment_contextual_mlm(
    tokens: List[str], 
    labels: List[str], 
    mlm_pipeline: Callable, 
    alpha: float = 0.1
) -> Tuple[List[str], List[str]]:
    """
    Contextual MLM Replacement. Masks a token and replaces it via the provided PLM.
    
    Args:
        tokens: Original token sequence.
        labels: Original BIO label sequence.
        mlm_pipeline: A HuggingFace pipeline or custom function. YOU MUST PROVIDE THIS MODEL.
                      Expected signature: mlm_pipeline(masked_sentence) -> predicted_token
        alpha: Probability of masking a non-entity token.
    """
    logger.info("Executing Contextual MLM replacement.")
    # Placeholder for actual MLM logic to be implemented by the user
    # ensuring no fabricated model calls are made.
    if mlm_pipeline is None:
        raise ValueError("mlm_pipeline cannot be None. Please instantiate and pass your MLM model.")
    
    aug_tokens = copy.deepcopy(tokens)
    o_indices = [i for i, label in enumerate(labels) if label == "O"]
    num_to_mask = max(1, int(alpha * len(o_indices)))
    target_indices = random.sample(o_indices, min(num_to_mask, len(o_indices)))

    for idx in target_indices:
        original_word = aug_tokens[idx]
        aug_tokens[idx] = "[MASK]"
        masked_sentence = " ".join(aug_tokens)
        
        # Invoke the injected PLM
        predicted_word = mlm_pipeline(masked_sentence)
        aug_tokens[idx] = predicted_word
        logger.debug(f"MLM Replacement: {original_word} -> {predicted_word}")

    return aug_tokens, labels


def augment_back_translation(
    tokens: List[str], 
    labels: List[str], 
    nmt_forward: Callable, 
    nmt_backward: Callable,
    alignment_heuristic: Callable
) -> Tuple[List[str], List[str]]:
    """
    Back Translation augmentation via NMT models, paired with an alignment heuristic.
    
    Args:
        tokens: Original token sequence.
        labels: Original BIO label sequence.
        nmt_forward: Function to translate EN -> Target. YOU MUST PROVIDE THIS MODEL.
        nmt_backward: Function to translate Target -> EN. YOU MUST PROVIDE THIS MODEL.
        alignment_heuristic: Function to realign translated tokens to original labels. 
                             YOU MUST PROVIDE THIS LOGIC.
    """
    logger.info("Executing Back Translation augmentation.")
    if not all([nmt_forward, nmt_backward, alignment_heuristic]):
         raise ValueError("Translation models and alignment heuristic must be provided.")
         
    sentence = " ".join(tokens)
    translated = nmt_forward(sentence)
    back_translated = nmt_backward(translated)
    
    aug_tokens = back_translated.split()
    aug_labels = alignment_heuristic(tokens, labels, aug_tokens)
    
    return aug_tokens, aug_labels


def augment_entity_aware(
    tokens: List[str], 
    labels: List[str], 
    entity_kb: Dict[str, List[List[str]]]
) -> Tuple[List[str], List[str]]:
    """
    Entity-Aware Replacement. Replaces an entity span with another of the same type.
    
    Args:
        tokens: Original token sequence.
        labels: Original BIO label sequence.
        entity_kb: Dictionary mapping entity types to a list of valid tokenized spans.
                   e.g., {"PER": [["John", "Doe"], ["Satya", "Nadella"]], "ORG": [["Google"]]}
                   YOU MUST PROVIDE THIS DATA.
    """
    from metrics import extract_entities_from_labels # Rely on your existing framework
    
    entities = extract_entities_from_labels(tokens, labels)
    if not entities:
        return tokens, labels
        
    # Pick a random entity span to replace
    target_entity = random.choice(entities)
    start_idx, end_idx, entity_type = target_entity
    
    if entity_type not in entity_kb or not entity_kb[entity_type]:
        logger.warning(f"Entity type {entity_type} not found in KB. Skipping replacement.")
        return tokens, labels
        
    replacement_span = random.choice(entity_kb[entity_type])
    
    # Construct new tokens
    aug_tokens = tokens[:start_idx] + replacement_span + tokens[end_idx+1:]
    
    # Construct new labels mapping the BIO format to the new span length
    if len(replacement_span) == 1:
        replacement_labels = [f"B-{entity_type}"]
    else:
        replacement_labels = [f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(replacement_span) - 1)
        
    aug_labels = labels[:start_idx] + replacement_labels + labels[end_idx+1:]
    
    logger.info(f"Entity-Aware Replacement: Swapped '{' '.join(tokens[start_idx:end_idx+1])}' with '{' '.join(replacement_span)}'")
    return aug_tokens, aug_labels