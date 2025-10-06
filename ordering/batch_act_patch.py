import torch
from nnsight import LanguageModel
from tqdm import trange
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
OUTPUT_DIR = Path("activation_patching_results")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / f'run_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model setup
logger.info("Loading Qwen-32B model...")
model = LanguageModel("Qwen/Qwen3-32B", device_map="auto")
logger.info(f"Model loaded. Config: {model.config.num_hidden_layers} layers")

ARTICLE_FIRST = [
    ['An artist Jane paints', 'The artist Jane paints'],
    ['The pet fish swims', 'A pet fish swims'],
    ['An animal fish swims', 'The animal fish swims'],
    ['A friend Tom visits', 'The friend Tom visits'],
    ['The woman Mary cooks', 'A woman Mary cooks'],
    ['A machine needs fuel', 'The machine needs fuel'],
    ['A toy is fun', 'The toy is fun'],
    ['A plant loves sun', 'The plant loves sun'],
    ['The dog is loud', 'A dog is loud'],
    ['A breeze blows softly', 'The breeze blows softly'],
    ['The mouse runs away', 'A mouse runs away'],
    ['A child plays happily', 'The child plays happily'],
    ['A clock ticks slowly', 'The clock ticks slowly'],
    ['A book smells fresh', 'The book smells fresh'],
    ['The library is peaceful', 'A library is peaceful'],
    ['A street sounds noisy', 'The street sounds noisy'],
    ['The train arrives soon', 'A train arrives soon'],
    ['A knife cuts easily', 'The knife cuts easily'],
    ['A flower smells sweet', 'The flower smells sweet'],
    ['A wire bends easily', 'The wire bends easily'],
]

PROMPT_TEMPLATE = """A ball rolls by. Q: What is the second word in the previous sentence? A: ball
Pizza is the best. Q: What is the first word in the previous sentence? A: Pizza
{}. Q: What is the first word in the previous sentence? A:"""

FEW_SHOT_EXAMPLES = """A ball rolls by. Q: What is the second word in the previous sentence? A: ball
Pizza is the best. Q: What is the first word in the previous sentence? A: Pizza\n"""
FSE_TOKENS = [tok.replace("Ġ", " ").replace("Ċ", "\n") for tok in model.tokenizer.tokenize(FEW_SHOT_EXAMPLES)]

QUESTION = "Q: What is the first word in the previous sentence? Answer in one word only!"
QUESTION_TOKENS = [tok.replace("Ġ", " ").replace("Ċ", "\n") for tok in model.tokenizer.tokenize(QUESTION)]

def get_clean_acts_logdiffs(
    model,
    clean_prompts,
    clean_targets,
    corrupt_targets
):
    """Batch clean run"""
    clean_activations = []
    clean_logit_diffs = []
    with model.trace(clean_prompts):
        # Get activations from all layers
        for l in range(model.config.num_hidden_layers):
            residual_output = model.model.layers[l].output[0].save()
            clean_activations.append(residual_output)

        # Get logit differences for all prompts
        logits = model.output.logits
        for i in range(len(clean_prompts)):
            clean_logit_diff = (
                logits[i, -1, clean_targets[i]] - logits[i, -1, corrupt_targets[i]]
            ).save()
            clean_logit_diffs.append(clean_logit_diff)

    torch.cuda.empty_cache()

    return clean_activations, clean_logit_diffs

def get_corrupt_logdiffs(
    model,
    corrupt_prompts,
    clean_targets,
    corrupt_targets,
):
    """Batch corrupt run"""
    corrupt_logit_diffs = []

    with model.trace(corrupt_prompts):
        logits = model.output.logits
        for i in range(len(corrupt_prompts)):
            corrupt_logit_diff = (
                logits[i, -1, clean_targets[i]] - logits[i, -1, corrupt_targets[i]]
            ).save()
            corrupt_logit_diffs.append(corrupt_logit_diff)

    torch.cuda.empty_cache()

    return corrupt_logit_diffs

def activation_patching(
    model,
    n_tokens,
    corrupt_prompts,
    clean_activations,
    clean_targets,
    corrupt_targets,
    corrupt_logit_diffs,
    total_logit_diffs,
):
    """Activation Patching"""
    accumulated_scores = []
    for l in trange(model.config.num_hidden_layers, desc="Layers"):
        layer_scores = []

        for t in range(n_tokens):
            
            # Batch all 20 prompts for this specific (layer, token) patch

            # Calculate noramlized scores for each prompt
            normalized_scores = []
            with model.trace(corrupt_prompts):
                # Patch the activation for all prompts at once
                for i in range(len(corrupt_prompts)):
                    model.model.layers[l].output[0][i, t, :] = clean_activations[l][i, t, :]
                
                # Get patched logits for all prompts
                patched_logits = model.output.logits

                for i in range(len(corrupt_prompts)):
                    patched_logit_diff = (
                        patched_logits[i, -1, clean_targets[i]] - patched_logits[i, -1, corrupt_targets[i]]
                    )

                    # Normalize
                    if total_logit_diffs[i] == 0:
                        normalized_score = 0.0
                    else:
                        normalized_score = (patched_logit_diff - corrupt_logit_diffs[i]) / total_logit_diffs[i]

                    normalized_scores.append(normalized_score.save())

            # Average across all 20 prompts for this (layer, token) position
            avg_score = sum([score.item() for score in normalized_scores]) / len(normalized_scores)
            layer_scores.append(avg_score)
            print(f"L{l}, T{t}, Avg_LD={avg_score:.4f}")

        torch.cuda.empty_cache()

        accumulated_scores.append(layer_scores)

    return accumulated_scores

def visualize_patching(
    model,
    accumulated_scores,
    clean_prompts,
):
    scores = np.array(accumulated_scores).T # Transpose so tokens are rows, layers are columns

    # Decode tokens from the first clean prompt (they're all the same structure)
    clean_input_ids = model.tokenizer.encode(clean_prompts[0])
    clean_tokens = [model.tokenizer.decode([token_id]) for token_id in clean_input_ids]

    # Create token labels
    tokens = ["few_shot_examples", "article", "sentence_tok_1", "sentence_tok_2", "sentence_tok_3",]
    for idx, clean_tok in enumerate(clean_tokens):
        if idx < (len(FSE_TOKENS) + 4):
            continue
        tokens.append(f'"{clean_tok}"')

    # Aggregate few-shot tokens
    num_fse_tokens = len(FSE_TOKENS)
    fse_aggregated = scores[:num_fse_tokens].mean(axis=0, keepdims=True)
    remaining_scores = scores[num_fse_tokens:]
    collapsed_scores = np.vstack([fse_aggregated, remaining_scores])

    print(f"Data range: min={collapsed_scores.min():.4f}, max={collapsed_scores.max():.4f}")

    # Plot
    plt.rcdefaults()
    with plt.rc_context(rc={"font.family": "Times New Roman", "font.size": 6}):
        fig, ax = plt.subplots(figsize=(6, len(tokens) * 0.08 + 1.8), dpi=200)
        
        heatmap = ax.pcolor(collapsed_scores, cmap="Purples", vmin=0, vmax=1)
        ax.invert_yaxis()
        
        # Y-axis: token labels
        ax.set_yticks([0.5 + i for i in range(len(tokens))])
        ax.set_yticklabels(tokens)
        
        # X-axis: layer labels
        num_layers = collapsed_scores.shape[1]
        tick_indices = np.arange(0, num_layers, 5)
        ax.set_xticks(tick_indices + 0.5)
        ax.set_xticklabels(tick_indices)
        
        ax.set_title("Indirect Effects of Residual Layers (Averaged over 20 prompts)")
        ax.set_xlabel("Layer")
        
        color_scale = plt.colorbar(heatmap)
        color_scale.ax.set_title("Normalized Score", y=-0.12, fontsize=8)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'qwen_32b_indirect_effects_heatmap_article_first.png', dpi=300, bbox_inches='tight')

def main():
    logger.info(f"Starting batch activation patching experiment at {timestamp}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    dataset = ARTICLE_FIRST
    batch_size=1

    # Prepare all prompts and target tokens
    clean_sentences = [pair[0] for pair in dataset]
    corrupt_sentences = [pair[1] for pair in dataset]

    clean_prompts = [PROMPT_TEMPLATE.format(sent) for sent in clean_sentences]
    corrupt_prompts = [PROMPT_TEMPLATE.format(sent) for sent in corrupt_sentences]

    # Extract target tokens (first word of each sentence, which is the article)
    clean_targets = [model.tokenizer.encode(sent.split()[0], add_special_tokens=False)[0]
                     for sent in clean_sentences]
    corrupt_targets = [model.tokenizer.encode(sent.split()[0], add_special_tokens=False)[0]
                       for sent in corrupt_sentences]
    
    # Get token length (should be the same for all)
    n_tokens = len(model.tokenizer.encode(clean_prompts[0]))
    print(f"Number of tokens per prompt: {n_tokens}")

    num_batches = (len(dataset) + batch_size - 1) // batch_size
    all_batch_scores = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_idx}-{end_idx})")
        
        # Get batch slices
        batch_clean_prompts = clean_prompts[start_idx:end_idx]
        batch_corrupt_prompts = corrupt_prompts[start_idx:end_idx]
        batch_clean_targets = clean_targets[start_idx:end_idx]
        batch_corrupt_targets = corrupt_targets[start_idx:end_idx]
        
        # Run clean and corrupt passes for this batch
        batch_clean_activations, batch_clean_logit_diffs = get_clean_acts_logdiffs(
            model, batch_clean_prompts, batch_clean_targets, batch_corrupt_targets
        )

        batch_corrupt_logit_diffs = get_corrupt_logdiffs(
            model, batch_corrupt_prompts, batch_clean_targets, batch_corrupt_targets
        )

        # Calculate total logit diff for each prompt in batch
        batch_total_logit_diffs = [
            (clean - corrupt).item() for clean, corrupt in 
            zip(batch_clean_logit_diffs, batch_corrupt_logit_diffs)
        ]
        
        # Run activation patching for this batch
        batch_scores = activation_patching(
            model=model,
            n_tokens=n_tokens,
            corrupt_prompts=batch_corrupt_prompts,
            clean_activations=batch_clean_activations,
            clean_targets=batch_clean_targets,
            corrupt_targets=batch_corrupt_targets,
            corrupt_logit_diffs=batch_corrupt_logit_diffs,
            total_logit_diffs=batch_total_logit_diffs
        )
        
        all_batch_scores.append(batch_scores)
        
        # Clear memory after each batch
        del batch_clean_activations, batch_clean_logit_diffs
        del batch_corrupt_logit_diffs, batch_total_logit_diffs
        torch.cuda.empty_cache()
    
    # Average scores across all batches
    # all_batch_scores is a list of [num_batches] x [num_layers] x [num_tokens]
    # We need to average across batches for each (layer, token) position
    num_layers = len(all_batch_scores[0])
    accumulated_scores = []
    
    for layer_idx in range(num_layers):
        layer_token_scores = []
        for token_idx in range(n_tokens):
            # Collect scores from all batches for this (layer, token)
            scores_across_batches = [
                batch_scores[layer_idx][token_idx] 
                for batch_scores in all_batch_scores
            ]
            # Average across all prompts
            avg_score = np.mean(scores_across_batches)
            layer_token_scores.append(avg_score)
        accumulated_scores.append(layer_token_scores)

    visualize_patching(
        model=model,
        accumulated_scores=accumulated_scores,
        clean_prompts=clean_prompts
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise