import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .config import ASSETS_DIR

# -------------------------
# Visualization Utilities
# -------------------------

def save_figure(filename):
    """
    Saves the current matplotlib figure with consistent professional settings.
    Saves to the global assets directory with 300 DPI resolution.
    """
    path = os.path.join(ASSETS_DIR, filename)
    
    # Ensure layout does not clip labels
    plt.tight_layout()
    
    # High resolution for documentation and reports
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"    Artifact Saved: {path}")
    
    plt.show()
    plt.close()

# -------------------------
# Training Diagnostics
# -------------------------

def plot_learning_curves(metrics, version="v1"):
    """
    Standardized learning curve plotter for loss and validation metrics.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Training Loss
    sns.lineplot(x=metrics["train_steps"], y=metrics["train_loss"], 
                 label='Training Loss', color='#4E79A7', linewidth=2.5)

    # Validation Loss (if available)
    if metrics["eval_loss"]:
        sns.lineplot(x=metrics["eval_steps"], y=metrics["eval_loss"], 
                     label='Validation Loss', color='#E15759', linewidth=2.5, marker='o')

    plt.title(f'Learning Curve: LyricLoop {version.upper()}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)

    save_figure(f"eval_loss_curve_{version}.png")

# -------------------------
# Confidence & Interpretability
# -------------------------

def plot_token_heatmap(token_conf_pairs, title="Confidence Heatmap", filename="heatmap.png"):
    """Draws a text heatmap where background color represents model confidence."""
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    x, y = 0.02, 0.85
    line_height = 0.12
    confidences = [p[1] for p in token_conf_pairs]
    avg_conf = np.mean(confidences) if confidences else 0

    ax.text(0.02, 0.95, f"{title} (Avg: {avg_conf:.2%})", 
            fontsize=12, fontweight='bold', transform=ax.transAxes)

    for t, score in token_conf_pairs:
        # Professional Color Scale: Green (High), Orange (Medium), Red (Low)
        if score > 0.7: bg = '#aaffaa'
        elif score > 0.3: bg = '#ffeeba'
        else: bg = '#ffcccc'

        clean_text = t.replace('\n', '↵ ')
        text_w = len(clean_text) * 0.015

        if x + text_w > 0.95:
            x = 0.02
            y -= line_height

        ax.text(x, y, clean_text, bbox=dict(facecolor=bg, edgecolor='none', pad=2, alpha=0.8),
                fontfamily='monospace', fontsize=10, transform=ax.transAxes)
        x += text_w + 0.005

    save_figure(filename)
    return avg_conf

def plot_confidence_summary(genres, scores, title="Confidence Summary", filename="conf_summary.png"):
    """Standardized bar chart for comparing confidence across genres."""
    plt.figure(figsize=(11, 6))
    x = np.arange(len(genres))
    width = 0.35
    palette = ['#A0A0A0', '#4E79A7', '#E15759']    # grey, blue, red

    if isinstance(scores, list):
        scores_dict = {"Model Output": scores}
        width = 0.5
    else:
        scores_dict = scores

    active_scores = {k: v for k, v in scores_dict.items() if len(v) == len(genres)}
    
    for i, (label, values) in enumerate(active_scores.items()):
        offset = (i - (len(active_scores)-1)/2) * width if len(active_scores) > 1 else 0
        bars = plt.bar(x + offset, values, width, label=label, 
                       color=palette[i % 3], edgecolor='black', alpha=0.8)

        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}', 
                     ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Average Confidence Score')
    plt.xticks(x, genres)
    plt.ylim(0, 1.1)
    if len(active_scores) > 1:
        plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    save_figure(filename)

# -------------------------
# Performance Comparison
# -------------------------

def plot_perplexity(genres, scores_dict, title="Model Perplexity", filename="perplexity.png", use_log=False):
    """Global plotter for perplexity scores with support for log-scaling."""
    plt.figure(figsize=(10, 6))
    if use_log: plt.yscale('log')

    x = np.arange(len(genres))
    comp_colors = ['#A0A0A0', '#4E79A7']    # grey for Baseline, blue for Fine-Tuned

    if len(scores_dict) == 1:
        label = list(scores_dict.keys())[0]
        values = list(scores_dict.values())[0]
        bars = plt.bar(genres, values, color='#A0A0A0', edgecolor='black', alpha=0.8)
    else:
        width = 0.35
        for i, (label, values) in enumerate(scores_dict.items()):
            offset = (i - (len(scores_dict)-1)/2) * width
            bars = plt.bar(x + offset, values, width, label=label, color=comp_colors[i % 2], edgecolor='black')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Perplexity (Lower is Better)', fontsize=12)
    plt.xticks(x, genres)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    if len(scores_dict) > 1: plt.legend()
    
    save_figure(filename)