from typing import List
import torch
from src.models import ModelandTokenizer

def category_to_indices(category: List[str], tokenizer) -> List[List[int]]:
    """
    Convert a list of words (category names) to their corresponding vocabulary indices.

    Args:
        category: List of words
        tokenizer: A model tokenizer for mapping words to their token indices

    Returns:
        List of vocabulary indices
    """
    return [tokenizer.encode(word, add_special_tokens=False) for word in category]