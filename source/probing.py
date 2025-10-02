import torch
import numpy as np
import matplotlib.pyplot as plt
from nnsight import LanguageModel
from dataclasses import dataclass, field
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

@dataclass
class Probe:
    positive_samples: List[str]
    negative_samples: List[str]
    model: LanguageModel
    seed: int = 9001

    # Use field for attributes that are initialized later
    positive_labels: np.ndarray = field(init=False)
    negative_labels: np.ndarray = field(init=False)
    y: np.array = field(init=False)
    train_idx: np.ndarray = field(init=False)
    test_idx: np.ndarray = field(init=False)
    n_layers: int = field(init=False)
    
    # Dictionaries to store activations, data, and trained probes by layer
    positive_activations: Dict[int, np.ndarray] = field(default_factory=dict)
    negative_activations: Dict[int, np.ndarray] = field(default_factory=dict)
    X_by_layer: Dict[int, np.ndarray] = field(default_factory=dict)
    probes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    scores: np.ndarray = field(default=None)

    def __post_init__(self):
        # FIX 1: Correctly get the number of layers
        self.n_layers = self.model.config.num_hidden_layers
        
        self.positive_labels = np.ones(len(self.positive_samples))
        self.negative_labels = np.zeros(len(self.negative_samples))
        self.y = np.concatenate([self.positive_labels, self.negative_labels])
        
        idx = np.arange(len(self.y))
        self.train_idx, self.test_idx = train_test_split(
            idx, test_size=0.2, stratify=self.y, random_state=self.seed
        )

    def collect_activations(self, token_idx: int = -1):
        print("Collecting activations...")
        
        # FIX 2: Initialize as a dictionary to hold lists of activations per layer
        pos_activations_by_layer = {i: [] for i in range(self.n_layers)}
        neg_activations_by_layer = {i: [] for i in range(self.n_layers)}

        for item in self.positive_samples:
            # FIX 3: Correct syntax for torch.no_grad
            with torch.no_grad():
                with self.model.trace(item):
                    for i, layer in enumerate(self.model.model.layers):
                        h = layer.output[0, token_idx, :].save()
                        pos_activations_by_layer[i].append(h)

        for item in self.negative_samples:
            with torch.no_grad():
                with self.model.trace(item):
                    for i, layer in enumerate(self.model.model.layers):
                        h = layer.output[0, token_idx, :].save()
                        neg_activations_by_layer[i].append(h)
        
        # Convert lists of tensors to stacked numpy arrays
        self.positive_activations = {
            k: np.stack([t.detach().cpu().numpy() for t in v], axis=0)
            for k, v in pos_activations_by_layer.items()
        }
        self.negative_activations = {
            k: np.stack([t.detach().cpu().numpy() for t in v], axis=0)
            for k, v in neg_activations_by_layer.items()
        }
        print("Activation collection complete.")
        return (self.positive_activations, self.negative_activations)
    
    def make_X_by_layer(self):
        self.X_by_layer = {
            k: np.concatenate(
                [
                    self.positive_activations[k],
                    self.negative_activations[k]
                ],
                axis=0
            ) for k in range(self.n_layers)
        }
        return self.X_by_layer

    def train(self):
        """A convenience method to run the full training pipeline."""
        self.collect_activations()
        self.make_X_by_layer()
        self.train_layer_probes()
        self.evaluate_probes()

    def train_layer_probes(self):
        print("Training probes for each layer...")
        probes = {}
        for layer in range(self.n_layers):
            # FIX 4: Use the correct attribute name
            X = self.X_by_layer[layer]
            
            X_train = X[self.train_idx]
            X_test = X[self.test_idx]
            y_train = self.y[self.train_idx]
            y_test = self.y[self.test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(
                class_weight='balanced',
                max_iter=1000, # Increased for better convergence
                solver='liblinear'
            )
            clf.fit(X_train_scaled, y_train)

            probes[layer] = {
                "model": clf,
                "scaler": scaler,
                "X_test": X_test_scaled, # Store the scaled version
                "y_test": y_test,
            }
        
        self.probes = probes
        print("Training complete.")
        
        # FIX 5: Return should be outside the loop
        return probes
        
    def evaluate_probes(self):
        if not self.probes:
            print("Probes not trained yet. Call train_layer_probes() first.")
            return

        print("Evaluating probes...")
        for layer, p in self.probes.items():
            y_score = p["model"].predict_proba(p["X_test"])[:, 1]
            auc_val = roc_auc_score(p["y_test"], y_score)
            print(f"Layer {layer}: AUC = {auc_val:.3f}")

    def score_by_layer(self, prompt: str, token_idx: int = -1):
        if not self.probes:
            raise ValueError("Probes must be trained before scoring.")

        hs = {}
        with torch.no_grad():
            with self.model.trace(prompt):
                for i, layer in enumerate(self.model.model.layers):
                    h = layer.output[0, token_idx, :].save()
                    hs[i] = h.detach().cpu().numpy()

        scores = []
        for layer in range(self.n_layers):
            x = hs[layer][None, :]
            scaler = self.probes[layer]['scaler']
            clf = self.probes[layer]['model']
            x_std = scaler.transform(x)
            
            s = float(clf.decision_function(x_std))
            scores.append(s)

        self.scores = np.array(scores)
        return self.scores

    def plot_layer_scores(self, title: str = "Concept Salience by Layer"):
        if self.scores is None:
            print("No scores to plot. Call score_by_layer() first.")
            return
            
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(self.scores)), self.scores, marker='o')
        plt.xlabel("Layer")
        plt.ylabel("Probe Score (Log-Odds)")
        plt.title(title)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()