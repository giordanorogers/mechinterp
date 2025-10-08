import torch
from nnsight import LanguageModel
from tqdm import trange
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

# Create output directory
OUTPUT_DIR = Path("activation_patching_results_article_second")
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

ARTICLE_SECOND = [
    ['Jack the boy runs', 'Jack a boy runs'],
    ['Jack a student learns', 'Jack the student learns'],
    ['Jack an actor plays', 'Jack the actor plays'],
    ['Jane the girl sings', 'Jane a girl sings'],
    ['Eat the pizza now', 'Eat a pizza now'],
    ['Jane an artist paints', 'Jane the artist paints'],
    ['Fish the pet swim', 'Fish a pet swims'],
    ['What the day brings', 'What a day brings'],
    ['Fish an animal swims', 'Fish the animal swims'],
    ['Tom the man works', 'Tom a man works'],
    ['Tom a friend visits', 'Tom the friend visits'],
    ['Tom an employee arrives', 'Tom the employee arrives'],
    ['Kate the woman cooks', 'Kate a woman cooks'],
    ['Kate a nurse cares', 'Kate the nurse cares'],
    ['Kate an author writes', 'Kate the author writes'],
    ['Cars the fast vehicle', 'Cars a fast vehicle'],
    ['Bob a scientist studies', 'Bob the scientist studies'],
    ['Cause a societal shift', 'Cause the societal shift'],
    ['See the root grow', 'See a root grow'],
    ['Trees a big plant', 'Trees the big plant']
]

PROMPT_TEMPLATE = """A ball rolls by. Q: What is the second word in the previous sentence? A: ball
Pizza is the best. Q: What is the first word in the previous sentence? A: Pizza
{}. Q: What is the second word in the previous sentence? A:"""

FEW_SHOT_EXAMPLES = """A ball rolls by. Q: What is the second word in the previous sentence? A: ball
Pizza is the best. Q: What is the first word in the previous sentence? A: Pizza\n"""
FSE_TOKENS = [tok.replace("Ġ", " ").replace("Ċ", "\n") for tok in model.tokenizer.tokenize(FEW_SHOT_EXAMPLES)]

def get_checkpoint_dir(batch_idx):
    """Get checkpoint directory for a specific batch"""
    checkpoint_dir = OUTPUT_DIR / "checkpoints" / f"batch_{batch_idx}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

def save_batch_metadata(batch_idx, metadata):
    """Save metadata about the batch (prompts, targets, etc.)"""
    checkpoint_dir = get_checkpoint_dir(batch_idx)
    with open(checkpoint_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata for batch {batch_idx}")

def save_clean_corrupt_data(batch_idx, clean_activations, clean_logit_diffs, 
                            corrupt_logit_diffs, total_logit_diffs):
    """Save clean and corrupt run data"""
    checkpoint_dir = get_checkpoint_dir(batch_idx)
    
    # Save activations (these are large)
    activation_file = checkpoint_dir / "clean_activations.pkl"
    with open(activation_file, 'wb') as f:
        pickle.dump([act.cpu() for act in clean_activations], f)
    
    # Save logit diffs
    logit_file = checkpoint_dir / "logit_diffs.pkl"
    with open(logit_file, 'wb') as f:
        pickle.dump({
            'clean_logit_diffs': [ld.cpu() for ld in clean_logit_diffs],
            'corrupt_logit_diffs': [ld.cpu() for ld in corrupt_logit_diffs],
            'total_logit_diffs': total_logit_diffs
        }, f)
    
    logger.info(f"Saved clean/corrupt data for batch {batch_idx}")

def load_clean_corrupt_data(batch_idx):
    """Load clean and corrupt run data if available"""
    checkpoint_dir = get_checkpoint_dir(batch_idx)
    activation_file = checkpoint_dir / "clean_activations.pkl"
    logit_file = checkpoint_dir / "logit_diffs.pkl"
    
    if not activation_file.exists() or not logit_file.exists():
        return None
    
    try:
        with open(activation_file, 'rb') as f:
            clean_activations = pickle.load(f)
        
        with open(logit_file, 'rb') as f:
            logit_data = pickle.load(f)
        
        logger.info(f"Loaded clean/corrupt data for batch {batch_idx}")
        return (clean_activations, 
                logit_data['clean_logit_diffs'],
                logit_data['corrupt_logit_diffs'],
                logit_data['total_logit_diffs'])
    except Exception as e:
        logger.warning(f"Failed to load checkpoint data for batch {batch_idx}: {e}")
        return None

def save_layer_results(batch_idx, layer_idx, layer_scores):
    """Save results for a specific layer"""
    checkpoint_dir = get_checkpoint_dir(batch_idx)
    layer_file = checkpoint_dir / f"layer_{layer_idx:03d}.npy"
    np.save(layer_file, np.array(layer_scores))
    logger.info(f"Saved layer {layer_idx} results for batch {batch_idx}")

def load_completed_layers(batch_idx, num_layers):
    """Load all completed layer results"""
    checkpoint_dir = get_checkpoint_dir(batch_idx)
    completed_layers = {}
    
    for layer_idx in range(num_layers):
        layer_file = checkpoint_dir / f"layer_{layer_idx:03d}.npy"
        if layer_file.exists():
            try:
                layer_scores = np.load(layer_file).tolist()
                completed_layers[layer_idx] = layer_scores
            except Exception as e:
                logger.warning(f"Failed to load layer {layer_idx}: {e}")
                break  # Stop at first failure to maintain order
    
    if completed_layers:
        logger.info(f"Loaded {len(completed_layers)} completed layers for batch {batch_idx}")
    
    return completed_layers

def save_batch_completion(batch_idx, batch_scores):
    """Mark batch as complete and save final results"""
    checkpoint_dir = get_checkpoint_dir(batch_idx)
    completion_file = checkpoint_dir / "batch_complete.npy"
    np.save(completion_file, np.array(batch_scores))
    logger.info(f"Marked batch {batch_idx} as complete")

def is_batch_complete(batch_idx):
    """Check if batch is already complete"""
    checkpoint_dir = get_checkpoint_dir(batch_idx)
    completion_file = checkpoint_dir / "batch_complete.npy"
    return completion_file.exists()

def load_completed_batch(batch_idx):
    """Load completed batch results"""
    checkpoint_dir = get_checkpoint_dir(batch_idx)
    completion_file = checkpoint_dir / "batch_complete.npy"
    
    if completion_file.exists():
        try:
            batch_scores = np.load(completion_file).tolist()
            logger.info(f"Loaded completed batch {batch_idx}")
            return batch_scores
        except Exception as e:
            logger.warning(f"Failed to load completed batch {batch_idx}: {e}")
    
    return None

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
    batch_idx,
    n_tokens,
    corrupt_prompts,
    clean_activations,
    clean_targets,
    corrupt_targets,
    corrupt_logit_diffs,
    total_logit_diffs,
):
    """Activation Patching with checkpointing"""
    num_layers = model.config.num_hidden_layers
    
    # Check for existing completed layers
    completed_layers = load_completed_layers(batch_idx, num_layers)
    
    accumulated_scores = []
    start_layer = 0
    
    # Resume from where we left off
    if completed_layers:
        start_layer = max(completed_layers.keys()) + 1
        accumulated_scores = [completed_layers[i] for i in range(len(completed_layers))]
        logger.info(f"Resuming from layer {start_layer}")
    
    for l in trange(start_layer, num_layers, desc="Layers"):
        layer_scores = []

        for t in range(n_tokens):
            # Calculate normalized scores for each prompt
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

            # Average across all prompts for this (layer, token) position
            avg_score = sum([score.item() for score in normalized_scores]) / len(normalized_scores)
            layer_scores.append(avg_score)
            print(f"L{l}, T{t}, Avg_LD={avg_score:.4f}")

        torch.cuda.empty_cache()

        accumulated_scores.append(layer_scores)
        
        # Save checkpoint after each layer
        save_layer_results(batch_idx, l, layer_scores)

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
    tokens = ["few_shot_examples", "sentence_tok_1", "article", "sentence_tok_3", "sentence_tok_4",]
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
        
        ax.set_title("Indirect Effects of Residuals (Article Second)")
        ax.set_xlabel("Layer")
        
        color_scale = plt.colorbar(heatmap)
        color_scale.ax.set_title("Normalized Score", y=-0.12, fontsize=8)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(OUTPUT_DIR / f'{timestamp}_indirect_effects_heatmap_article_second.png', 
                   dpi=300, bbox_inches='tight')

def main():
    logger.info(f"Starting batch activation patching experiment at {timestamp}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    dataset = ARTICLE_SECOND
    batch_size = 1

    # Prepare all prompts and target tokens
    clean_sentences = [pair[0] for pair in dataset]
    corrupt_sentences = [pair[1] for pair in dataset]

    clean_prompts = [PROMPT_TEMPLATE.format(sent) for sent in clean_sentences]
    corrupt_prompts = [PROMPT_TEMPLATE.format(sent) for sent in corrupt_sentences]

    # Extract target tokens (second word of each sentence, which is the article)
    clean_targets = [model.tokenizer.encode(sent.split()[1], add_special_tokens=False)[0]
                     for sent in clean_sentences]
    corrupt_targets = [model.tokenizer.encode(sent.split()[1], add_special_tokens=False)[0]
                       for sent in corrupt_sentences]
    
    # Get token length (should be the same for all)
    n_tokens = len(model.tokenizer.encode(clean_prompts[0]))
    print(f"Number of tokens per prompt: {n_tokens}")

    num_batches = (len(dataset) + batch_size - 1) // batch_size
    all_batch_scores = []

    for batch_idx in range(num_batches):
        # Check if batch is already complete
        completed_batch = load_completed_batch(batch_idx)
        if completed_batch is not None:
            logger.info(f"Batch {batch_idx + 1}/{num_batches} already complete, skipping")
            all_batch_scores.append(completed_batch)
            continue
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (prompts {start_idx}-{end_idx})")
        
        # Get batch slices
        batch_clean_prompts = clean_prompts[start_idx:end_idx]
        batch_corrupt_prompts = corrupt_prompts[start_idx:end_idx]
        batch_clean_targets = clean_targets[start_idx:end_idx]
        batch_corrupt_targets = corrupt_targets[start_idx:end_idx]
        
        # Save metadata
        metadata = {
            'batch_idx': batch_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n_tokens': n_tokens,
            'timestamp': timestamp
        }
        save_batch_metadata(batch_idx, metadata)
        
        # Try to load existing clean/corrupt data
        loaded_data = load_clean_corrupt_data(batch_idx)
        
        if loaded_data is not None:
            logger.info(f"Using cached clean/corrupt data for batch {batch_idx}")
            batch_clean_activations, batch_clean_logit_diffs, batch_corrupt_logit_diffs, batch_total_logit_diffs = loaded_data
        else:
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
            
            # Save for future use
            save_clean_corrupt_data(batch_idx, batch_clean_activations, batch_clean_logit_diffs,
                                   batch_corrupt_logit_diffs, batch_total_logit_diffs)
        
        # Run activation patching for this batch (with checkpointing)
        batch_scores = activation_patching(
            model=model,
            batch_idx=batch_idx,
            n_tokens=n_tokens,
            corrupt_prompts=batch_corrupt_prompts,
            clean_activations=batch_clean_activations,
            clean_targets=batch_clean_targets,
            corrupt_targets=batch_corrupt_targets,
            corrupt_logit_diffs=batch_corrupt_logit_diffs,
            total_logit_diffs=batch_total_logit_diffs
        )
        
        # Mark batch as complete
        save_batch_completion(batch_idx, batch_scores)
        
        all_batch_scores.append(batch_scores)
        
        # Clear memory after each batch
        del batch_clean_activations, batch_clean_logit_diffs
        del batch_corrupt_logit_diffs, batch_total_logit_diffs
        torch.cuda.empty_cache()
    
    # Average scores across all batches
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

    # Save final aggregated results
    final_results_file = OUTPUT_DIR / f'{timestamp}_final_results.npy'
    np.save(final_results_file, np.array(accumulated_scores))
    logger.info(f"Saved final results to {final_results_file}")

    visualize_patching(
        model=model,
        accumulated_scores=accumulated_scores,
        clean_prompts=clean_prompts
    )
    
    logger.info("Experiment complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise