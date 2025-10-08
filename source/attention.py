import torch
import numpy as np
from scipy import stats
from pathlib import Path
from tqdm.auto import tqdm
from nnsight import LanguageModel
from typing import Optional, List, Dict, Tuple

from source.utils import generate_text_hash, get_cache_path, clear_gpu_memory
from source.tokens import get_raw_tokens, get_sentence_token_boundaries

def remove_and_renormalize_bos(
    attn_matrix: np.ndarray,
    bos_idx: int = 0,
    exclude_from_bos: bool = True
) -> np.ndarray:
    """
    Remove BOS token and renormalize attention weights.
    
    Args:
        attn_matrix: Attention matrix of shape (seq_len, seq_len) or (n_sentences, n_sentences)
        bos_idx: Index of BOS token (default: 0)
        exclude_from_bos: Whether to also zero out attention FROM the BOS token
    
    Returns:
        Renormalized attention matrix with BOS removed
    """
    matrix = attn_matrix.copy()
    
    # Zero out attention TO BOS
    matrix[:, bos_idx] = 0
    
    # Optionally zero out attention FROM BOS
    if exclude_from_bos:
        matrix[bos_idx, :] = 0
    
    # Renormalize each row to sum to 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums
    
    return matrix

def extract_attention_and_logits(
    model,
    tokenizer,
    text,
    model_name: str = "unknown",
    return_logits: bool = True,
    attn_layers: Optional[List[int]] = None,
    token_range_to_mask: Optional[List[int]] = None,
    mask_layers: Optional[Dict[int, List[int]]] = None,
):
    """
    Run a forward pass to extract attention weights and logits.
    """

    logits = None
    attention_weights = {}

    try:
        with torch.no_grad():
            with model.trace(
                text,
                output_attentions=True,
            ):
                output = model.model.output.save()
                logits = model.lm_head(output.last_hidden_state).save()

            if hasattr(output, "attentions"):
                for layer_idx, attn_weights in enumerate(output.attentions):
                    if attn_layers is not None and layer_idx not in attn_layers:
                        continue
                    attention_weights[layer_idx] = attn_weights.detach().cpu()
            else:
                raise ValueError("No attentions in output")
        
    except Exception as e:
        print(f"ERROR: {e}")

    all_tokens = tokenizer(text).input_ids
    token_texts = tokenizer.convert_ids_to_tokens(all_tokens)

    result = {
        "text": text,
        "tokens": all_tokens,
        "token_texts": token_texts,
        "input_length": len(all_tokens),
        "attention_weights": attention_weights,
    }

    if logits is not None:
        result['logits'] = logits

    return result

def analyze_text(
    text: str,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    return_logits: bool = False,
    attn_layers: Optional[List[int]] = None,
    token_range_to_mask: Optional[List[int]] = None,
    layers_to_mask: Optional[Dict[int, List[int]]] = None,
    device_map: str = "auto"
):
    """
    Analyze a text using a model's forward pass.
    """
    model = LanguageModel(model_name, device_map=device_map, attn_implementation="eager")
    tokenizer = model.tokenizer

    result = extract_attention_and_logits(
        model,
        tokenizer,
        text,
        model_name=model_name,
        return_logits=return_logits,
        attn_layers=attn_layers,
        token_range_to_mask=token_range_to_mask,
        mask_layers=layers_to_mask,
    )

    del model
    clear_gpu_memory()

    return result

def get_attention_matrix(
    text: str,
    model_name: str,
    layer: int,
    head: int,
    device_map: str = "auto"
):
    """
    Get the attention matrix for a specific layer and head for given text.
    """
    result = analyze_text(
        text=text,
        model_name=model_name,
        return_logits=False,
        attn_layers=None,
        device_map=device_map,
    )

    if len(result['attention_weights']) == 0:
        raise ValueError("No attention weights returned")
    
    matrix = result['attention_weights'][layer][0, head].numpy().astype(np.float32)
    return matrix

def compute_averaged_matrix(
    attn_matrix: np.ndarray,
    sentence_boundaries: List[Tuple[int, int]],
    remove_bos: bool = True,
    bos_idx: int = 0
):
    """
    Compute averaged attention matrix from raw matrix and boundaries.
    
    Args:
        attn_matrix: Raw token-level attention matrix
        sentence_boundaries: List of (start, end) token boundaries for each sentence
        remove_bos: Whether to remove and renormalize BOS attention
        bos_idx: Index of the BOS token (default: 0)
    
    Returns:
        Averaged attention matrix at sentence level
    """
    n = len(sentence_boundaries)
    result = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        row_start, row_end = sentence_boundaries[i]
        row_start = min(row_start, attn_matrix.shape[0] - 1)
        row_end = min(row_end, attn_matrix.shape[0] - 1)

        if row_start >= row_end:
            continue

        for j in range(n):
            col_start, col_end = sentence_boundaries[j]
            col_start = min(col_start, attn_matrix.shape[1] - 1)
            col_end = min(col_end, attn_matrix.shape[1] - 1)

            if col_start >= col_end:
                continue

            region = attn_matrix[row_start:row_end, col_start:col_end]
            if region.size > 0:
                result[i, j] = np.mean(region)
    
    # Remove BOS and renormalize after averaging
    if remove_bos:
        result = remove_and_renormalize_bos(result, bos_idx=bos_idx)

    return result

def compute_all_attention_matrices(
    text: str,
    model_name: str,
    sentences: Optional[List[str]],
    cache_dir: str = "avg_matrix",
    text_id: Optional[str] = None,
    device_map: str = "auto",
    force_recompute: bool = False,
    verbose: bool = True,
    remove_bos: bool = True,
) -> bool:
    """
    Compute attention matrices for all layers and heads at once.
    This is more efficient than computing them one by one.
    
    Args:
        remove_bos: Whether to remove and renormalize BOS attention
    """
    model = LanguageModel(model_name)

    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    if cache_dir and not text_id:
        text_id = generate_text_hash(text, sentences)

    tokens = get_raw_tokens(
        model_name,
        text,
    )

    result = analyze_text(
        text,
        model_name=model_name,
        attn_layers=None,
        return_logits=False,
        device_map=device_map
    )

    sentence_boundaries = None
    if sentences:
        sentence_boundaries = get_sentence_token_boundaries(
            text,
            sentences,
            model_name
        )

    for layer in tqdm(range(n_layers), desc="Saving avg. matrices"):
        for head in range(n_heads):
            matrix = result['attention_weights'][layer][0, head].numpy().astype(np.float32)

            if sentence_boundaries:
                matrix = compute_averaged_matrix(
                    matrix, 
                    sentence_boundaries,
                    remove_bos=remove_bos
                )
            elif remove_bos:
                # Even without sentence boundaries, renormalize token-level attention
                matrix = remove_and_renormalize_bos(matrix)
            
            if cache_dir and text_id:
                cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, matrix)

    if verbose:
        print(f"Saved to {cache_path}")

    return text_id

def get_avg_attention_matrix(
    text: str,
    model_name: str,
    layer: int,
    head: int,
    sentences: Optional[List[str]],
    device_map: str = "auto",
    cache_dir: Optional[str] = "attn_cache",
    text_id: Optional[str] = None,
    force_recompute: bool = False,
    remove_bos: bool = True,
) -> np.ndarray:
    """
    Get averaged attention matrix for a specific layer and head.
    
    Args:
        remove_bos: Whether to remove and renormalize BOS attention
    """
    if cache_dir and not text_id:
        text_id = generate_text_hash(text, sentences)
    
    if cache_dir and text_id and not force_recompute:
        cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
        if Path(cache_path).exists():
            return np.load(cache_path)
    
    # If not in cache, compute it
    matrix = get_attention_matrix(text, model_name, layer, head, device_map)

    if sentences is None:
        # Token-level matrix
        if remove_bos:
            matrix = remove_and_renormalize_bos(matrix)
        result = matrix
    else:
        # Sentence-level matrix
        sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)
        result = compute_averaged_matrix(
            matrix, 
            sentence_boundaries,
            remove_bos=remove_bos
        )

    if cache_dir and text_id:
        cache_path = get_cache_path(cache_dir, text_id, model_name, layer, head)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, result)
    
    return result

def get_vertical_scores(
    avg_matrix: np.ndarray,
    proximity_ignore: int = 20,
    control_depth: bool = True,
    score_type: str = "mean",
    bos_already_removed: bool = True,
) -> np.ndarray:
    """
    Calculate vertical attention scores from an averaged attention matrix.
    
    Args:
        avg_matrix: Averaged attention matrix (should already have BOS removed if desired)
        proximity_ignore: Number of nearby positions to ignore
        control_depth: Whether to rank-normalize by depth
        score_type: "mean" or "median"
        bos_already_removed: Whether BOS has already been removed from avg_matrix
    
    Returns:
        Array of vertical attention scores for each position
    """
    # Clean the matrix - set upper triangle to NaN
    n = avg_matrix.shape[0]
    trius = np.triu_indices_from(avg_matrix, k=1)

    avg_mat = avg_matrix.copy()
    avg_mat[trius] = np.nan

    # Ignore nearby positions
    trils = np.triu_indices_from(
        avg_mat, k=-proximity_ignore + 1
    )
    avg_mat[trils] = np.nan

    # Optionally exclude BOS if not already removed
    if not bos_already_removed:
        avg_mat[:, 0] = np.nan
        avg_mat[0, :] = np.nan

    # Rank normalization by depth
    if control_depth:
        per_row = np.sum(~np.isnan(avg_mat), axis=1)
        # Avoid division by zero
        per_row[per_row == 0] = 1
        avg_mat = stats.rankdata(avg_mat, axis=1, nan_policy="omit") / per_row[:, None]

    n = avg_mat.shape[-1]
    vert_scores = []

    for i in range(n):
        vert_lines = avg_mat[i + proximity_ignore:, i]

        if score_type == "mean":
            vert_score = np.nanmean(vert_lines)
        elif score_type == "median":
            vert_score = np.nanmedian(vert_lines)
        else:
            raise ValueError(f"Unknown score_type: {score_type}")
        
        vert_scores.append(vert_score)  # Fixed typo: was vert_scores

    return np.array(vert_scores)

def get_attention_to_step(
    text: str,
    model_name: str,
    layer: int,
    head: int,
    step_idx: int,
    sentences: List[str],
    device_map: str = "auto",
    cache_dir: Optional[str] = "attn_cache",
    remove_bos: bool = True,
) -> np.ndarray:
    """
    Get attention from all tokens to a specific step/sentence.
    
    Args:
        remove_bos: Whether to remove and renormalize BOS attention
    """
    avg_matrix = get_avg_attention_matrix(
        text,
        model_name,
        layer,
        head,
        sentences,
        device_map,
        cache_dir=cache_dir,
        remove_bos=remove_bos,
    )

    return avg_matrix[:, step_idx]