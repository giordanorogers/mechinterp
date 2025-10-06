"""
Batch Activation Patching Script for Article Position Analysis
Run with: python batch_activation_patching.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import torch
from datetime import datetime
import pickle
import json
from pathlib import Path

from nnsight import LanguageModel

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

# Constants
PROMPT_TEMPLATE = """A ball rolls by. Q: What is the second word in the previous sentence? A: ball
Pizza is the best. Q: What is the first word in the previous sentence? A: Pizza
{}. Q: What is the {} word in the previous sentence? Answer in one word only! A:"""
OPTIONS = ["first", "second", "third"]
FEW_SHOT_EXAMPLES = """A ball rolls by. Q: What is the second word in the previous sentence? A: ball
Pizza is the best. Q: What is the first word in the previous sentence? A: Pizza"""

# Get FSE tokens
FSE_TOKENS = [tok.replace("Ġ", " ").replace("Ċ", "\n") 
              for tok in model.tokenizer.tokenize(FEW_SHOT_EXAMPLES)]

# Datasets
article_first = [
    ['An artist Jane paints', 'The artist Jane paints'],
    ['The pet fish swims', 'A pet fish swims'],
    ['An animal fish swims', 'The animal fish swims'],
    ['A friend Tom visits', 'The friend Tom visits'],
    ['The woman Mary cooks', 'A woman Mary cooks'],
    ['A machine needs fuel', 'The machine needs fuel'],
    ['An invention changed society', 'The invention changed society'],
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

article_second = [
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

article_third = [
    ['Jack is a boy', 'Jack is the boy'],
    ['Jane is the girl', 'Jane is a girl'],
    ['Fish are the pet', 'Fish are a pet'],
    ['Fish need a bowl', 'Fish need the bowl'],
    ['Tom is the man', 'Tom is a man'],
    ['Jim is a guy', 'Jim is the guy'],
    ['Bob is an friend', 'Bob is the friend'],
    ['Mary is the lady', 'Mary is a lady'],
    ['Jane is a woman', 'Jane is the woman'],
    ['Cars are the machine', 'Cars are a machine'],
    ['I want a flower', 'I want the flower'],
    ['Weapons are the menace', 'Weapons are a menace'],
    ['I had an idea', 'I had the idea'],
    ['I love a forest', 'I love the forest'],
    ['I love the girl', 'I love a girl'],
    ['Kids play a game', 'Kids play the game'],
    ['He wants the drink', 'He wants a drink'],
    ['She says the word', 'She says a word'],
    ['I know a way', 'I know the way'],
    ['Movie of the era', 'Movie of an era']
]


def run_batch_activation_patching(dataset, article_position, dataset_name, batch_size=4):
    """Run batch activation patching for a dataset."""
    
    logger.info(f"Starting {dataset_name}...")
    
    # Prepare prompts and targets
    clean_sentences = [pair[0] for pair in dataset]
    corrupt_sentences = [pair[1] for pair in dataset]
    
    clean_prompts = [PROMPT_TEMPLATE.format(sent, OPTIONS[article_position]) for sent in clean_sentences]
    corrupt_prompts = [PROMPT_TEMPLATE.format(sent, OPTIONS[article_position]) for sent in corrupt_sentences]
    
    # Extract target tokens
    clean_targets = [model.tokenizer.encode(" " + sent.split()[article_position], add_special_tokens=False)[0] 
                     for sent in clean_sentences]
    corrupt_targets = [model.tokenizer.encode(" " + sent.split()[article_position], add_special_tokens=False)[0] 
                       for sent in corrupt_sentences]
    
    n_tokens = len(model.tokenizer.encode(clean_prompts[0]))
    logger.info(f"Tokens per prompt: {n_tokens}")

    n_samples = len(dataset)
    n_batches = (n_samples + batch_size - 1) // batch_size  # FIXED: Added parentheses
    
    logger.info(f"Processing {n_samples} samples in {n_batches} batches of size {batch_size}")

    all_clean_activations = []
    all_clean_logit_diffs = []
    all_corrupt_logit_diffs = []
    
    # STEP 1 & 2: Process clean and corrupt runs in batches
    for batch_idx in range(n_batches):  # FIXED: Was n_batchs
        logger.info(f"Processing batch {batch_idx + 1}/{n_batches}")
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)

        batch_clean_prompts = clean_prompts[start_idx:end_idx]
        batch_corrupt_prompts = corrupt_prompts[start_idx:end_idx]
        batch_clean_targets = clean_targets[start_idx:end_idx]
        batch_corrupt_targets = corrupt_targets[start_idx:end_idx]

        # Step 1: Clean run for this batch
        batch_clean_acts = []
        batch_clean_ld = []
        with model.trace(batch_clean_prompts):
            for l in range(model.config.num_hidden_layers):
                residual_output = model.model.layers[l].output[0].save()
                batch_clean_acts.append(residual_output)

            logits = model.output.logits
            for i in range(len(batch_clean_prompts)):
                clean_logit_diff = (
                    logits[i, -1, batch_clean_targets[i]] - 
                    logits[i, -1, batch_corrupt_targets[i]]
                ).save()
                batch_clean_ld.append(clean_logit_diff)

        # Move to CPU to free GPU memory
        batch_clean_acts = [[act[i].cpu() for i in range(len(batch_clean_prompts))]
                            for act in batch_clean_acts]
        batch_clean_ld = [ld.cpu() for ld in batch_clean_ld]

        all_clean_activations.append(batch_clean_acts)
        all_clean_logit_diffs.extend(batch_clean_ld)

        # Step 2: Corrupt run for this batch
        batch_corrupt_ld = []
        with model.trace(batch_corrupt_prompts):
            logits = model.output.logits
            for i in range(len(batch_corrupt_prompts)):
                corrupt_logit_diff = (
                    logits[i, -1, batch_clean_targets[i]] -
                    logits[i, -1, batch_corrupt_targets[i]]
                ).save()
                batch_corrupt_ld.append(corrupt_logit_diff)

        all_corrupt_logit_diffs.extend([ld.cpu() for ld in batch_corrupt_ld])

        torch.cuda.empty_cache()
    
    # Restructure activations: [layer][sample] instead of [batch][layer][sample]
    logger.info("Restructuring activations...")
    clean_activations = []
    for l in range(model.config.num_hidden_layers):
        layer_acts = []
        for batch_acts in all_clean_activations:
            layer_acts.extend(batch_acts[l])
        clean_activations.append(layer_acts)
    
    # Compute total logit diffs
    total_logit_diffs = [clean - corrupt for clean, corrupt in 
                         zip(all_clean_logit_diffs, all_corrupt_logit_diffs)]
    

    # Get device for each layer
    logger.info("Determining layer device placement...")
    layer_devices = {}
    for l in range(model.config.num_hidden_layers):
        layer_devices[l] = next(model.model.layers[l].parameters()).device
    logger.info(f"Layers distributed across devices: {set(layer_devices.values())}")
    
    # STEP 3: Activation Patching (ONE SAMPLE AT A TIME)
    logger.info("Running activation patching interventions...")
    accumulated_scores = []
    
    for l in trange(model.config.num_hidden_layers, desc=f"{dataset_name}"):
        layer_device = layer_devices[l]
        layer_scores = []
        
        for t in range(n_tokens):
            # Process ONE SAMPLE AT A TIME to minimize memory usage
            all_normalized_scores = []
            
            for sample_idx in range(n_samples):
                torch.cuda.empty_cache()  # Clear before each sample
                
                with model.trace([corrupt_prompts[sample_idx]]):
                    # Patch activation for this single sample
                    model.model.layers[l].output[0][0, t, :] = clean_activations[l][sample_idx][t, :].to(layer_device)
                    
                    patched_logits = model.output.logits
                    
                    patched_logit_diff = (
                        patched_logits[0, -1, clean_targets[sample_idx]] - 
                        patched_logits[0, -1, corrupt_targets[sample_idx]]
                    )
                    
                    if total_logit_diffs[sample_idx] == 0:
                        normalized_score = 0.0
                    else:
                        normalized_score = ((patched_logit_diff - all_corrupt_logit_diffs[sample_idx]) / 
                                          total_logit_diffs[sample_idx])
                    
                    all_normalized_scores.append(normalized_score.save())
            
            avg_score = sum([score.item() for score in all_normalized_scores]) / len(all_normalized_scores)
            layer_scores.append(avg_score)
        
        accumulated_scores.append(layer_scores)
        
        # More aggressive cache clearing
        torch.cuda.empty_cache()
        
        # Save checkpoint every 10 layers
        if (l + 1) % 10 == 0:
            checkpoint_path = OUTPUT_DIR / f"{dataset_name.replace(' ', '_')}_checkpoint_layer_{l+1}_{timestamp}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(accumulated_scores, f)
            logger.info(f"Checkpoint saved at layer {l+1}")
    
    return np.array(accumulated_scores), clean_prompts[0]


def create_token_labels(clean_prompt, article_position):
    """Create token labels with placeholders."""
    clean_input_ids = model.tokenizer.encode(clean_prompt)
    tokens = ["few_shot_examples"]
    sentence_tok_counter = 1
    
    for idx in range(len(FSE_TOKENS), len(clean_input_ids)):
        sentence_position = idx - len(FSE_TOKENS)
        
        if sentence_position == article_position:
            tokens.append("article")
        elif sentence_position < 4:
            tokens.append(f"sentence_tok_{sentence_tok_counter}")
            sentence_tok_counter += 1
        elif sentence_position == 4:
            tokens.append("question")
        else:
            tokens.append("last_token")
    
    return tokens


def plot_and_save_results(scores, tokens, dataset_name, article_position):
    """Plot and save results."""
    scores = scores.T
    
    # Aggregate FSE tokens
    num_fse_tokens = len(FSE_TOKENS)
    fse_aggregated = scores[:num_fse_tokens].mean(axis=0, keepdims=True)
    remaining_scores = scores[num_fse_tokens:]
    collapsed_scores = np.vstack([fse_aggregated, remaining_scores])
    
    logger.info(f"{dataset_name} - Data range: min={collapsed_scores.min():.4f}, max={collapsed_scores.max():.4f}")
    
    # Plot
    plt.rcdefaults()
    with plt.rc_context(rc={"font.family": "Times New Roman", "font.size": 6}):
        fig, ax = plt.subplots(figsize=(6, len(tokens) * 0.08 + 1.8), dpi=200)
        
        heatmap = ax.pcolor(collapsed_scores, cmap="Purples", vmin=0, vmax=1)
        ax.invert_yaxis()
        
        ax.set_yticks([0.5 + i for i in range(len(tokens))])
        ax.set_yticklabels(tokens)
        
        num_layers = collapsed_scores.shape[1]
        tick_indices = np.arange(0, num_layers, 5)
        ax.set_xticks(tick_indices + 0.5)
        ax.set_xticklabels(tick_indices)
        
        ax.set_title(f"Indirect Effects - {dataset_name} (Article at position {article_position})")
        ax.set_xlabel("Layer")
        
        color_scale = plt.colorbar(heatmap)
        color_scale.ax.set_title("Normalized Score", y=-0.12, fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = OUTPUT_DIR / f"{dataset_name.replace(' ', '_')}_{timestamp}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_path}")
        
        plt.close()


def main():
    """Main execution function."""
    
    logger.info(f"Starting batch activation patching experiment at {timestamp}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    datasets = [
        (article_first, 0, "Article_First"),
        (article_second, 1, "Article_Second"),
        (article_third, 2, "Article_Third")
    ]
    
    all_results = {}
    
    for dataset, article_pos, name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {name}")
        logger.info(f"Article position: {article_pos}")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Run activation patching
            scores, sample_prompt = run_batch_activation_patching(dataset, article_pos, name)
            
            # Create labels
            token_labels = create_token_labels(sample_prompt, article_pos)
            
            # Save raw results
            results_dict = {
                'scores': scores,
                'token_labels': token_labels,
                'article_position': article_pos,
                'dataset': dataset,
                'timestamp': timestamp
            }
            
            results_path = OUTPUT_DIR / f"{name}_results_{timestamp}.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results_dict, f)
            logger.info(f"Results saved to {results_path}")
            
            # Plot and save
            plot_and_save_results(scores, token_labels, name, article_pos)
            
            all_results[name] = results_dict
            logger.info(f"Completed {name}\n")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}", exc_info=True)
            continue
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'model': "Qwen/Qwen2.5-32B-Instruct",
        'num_layers': model.config.num_hidden_layers,
        'datasets_processed': list(all_results.keys()),
        'output_dir': str(OUTPUT_DIR)
    }
    
    summary_path = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Experiment completed successfully!")
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"All outputs in: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise