import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

def plot_sentence_attention(
    model_name,
    matrix: Optional[str] = None,
    text_id: Optional[str] = None,
    layer: Optional[int] = None,
    head: Optional[int] = None,
    sentences: Optional[List[str]] = None,
    n_sentences: Optional[int] = None,
    cache_dir: str = "avg_matrix",
    cmap: str = "Blues",
    figsize: Tuple[int] = (22, 18)
):
    
    if matrix is None and text_id:
        matrix = np.load(
            f"{cache_dir}/{model_name}/{text_id}/{layer}_{head}.npy"
        )
    elif matrix is None and text_id is None:
        raise ValueError("Must provide a matrix or the path information.")

    plt.figure(figsize=figsize)

    plt.imshow(
        matrix,
        cmap=cmap,
        aspect="auto"
    )
    plt.colorbar(
        label="Attention Weight"
    )
    plt.xlabel(
        'Desitination Sentence'
    )
    plt.ylabel(
        'Source Sentence'
    )

    num_sentences = 0
    if sentences:
        num_sentences = len(sentences)
    elif n_sentences:
        num_sentences = n_sentences
    else:
        raise ValueError("Must pass either `sentences` or `n_sentences`.")
    plt.xticks(
        ticks=[i for i in range(num_sentences)],
        labels=[f'{i+1}' for i in range(num_sentences)]
    )
    plt.yticks(
        ticks=[i for i in range(num_sentences)],
        labels=[f'{i+1}' for i in range(num_sentences)]
    )

    if layer and head:
        plt.title(
            f'Layer {layer}, Head {head}'
        )
    else:
        plt.title("")

    plt.show()
