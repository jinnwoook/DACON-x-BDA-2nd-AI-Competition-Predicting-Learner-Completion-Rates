#!/usr/bin/env python3
"""Generate architecture and pipeline diagrams for README."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def create_pipeline_diagram():
    """Create the main pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    colors = {
        'data': '#E3F2FD',
        'preprocess': '#FFF3E0',
        'model': '#E8F5E9',
        'ensemble': '#FCE4EC',
        'output': '#F3E5F5',
        'border': '#37474F'
    }

    # Title
    ax.text(7, 9.5, 'BDA Competition - Ensemble Pipeline Architecture',
            fontsize=16, fontweight='bold', ha='center', va='center',
            color='#1565C0')

    # Stage 1: Data
    box1 = FancyBboxPatch((0.5, 7), 3, 1.5, boxstyle="round,pad=0.05",
                          facecolor=colors['data'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box1)
    ax.text(2, 7.75, 'Raw Data', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(2, 7.35, 'train.csv / test.csv', fontsize=9, ha='center', va='center', style='italic')

    # Stage 2: Preprocessing
    box2 = FancyBboxPatch((4.5, 7), 5, 1.5, boxstyle="round,pad=0.05",
                          facecolor=colors['preprocess'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box2)
    ax.text(7, 7.75, 'Text Preprocessing', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(7, 7.35, 'Cleaning + Tokenization + BERT Encoding', fontsize=9, ha='center', va='center', style='italic')

    # Arrow 1->2
    ax.annotate('', xy=(4.4, 7.75), xytext=(3.6, 7.75),
                arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))

    # Stage 3: Models (multiple boxes)
    model_names = [
        ('BERT\nClassifier', 0.5),
        ('SimCSE\nBERT', 3.2),
        ('Enhanced\nBERT', 5.9),
        ('Meta\nLearner', 8.6),
        ('Prob\nAverage', 11.3)
    ]

    for name, x in model_names:
        box = FancyBboxPatch((x, 4.5), 2.2, 1.8, boxstyle="round,pad=0.05",
                             facecolor=colors['model'], edgecolor=colors['border'], linewidth=2)
        ax.add_patch(box)
        ax.text(x + 1.1, 5.4, name, fontsize=9, fontweight='bold', ha='center', va='center')

    # Arrows from preprocessing to models
    for name, x in model_names:
        ax.annotate('', xy=(x + 1.1, 6.3), xytext=(7, 6.9),
                    arrowprops=dict(arrowstyle='->', color=colors['border'], lw=1.5,
                                  connectionstyle='arc3,rad=0'))

    # Stage 4: 10 Ensemble Files
    box4 = FancyBboxPatch((2, 2.2), 10, 1.5, boxstyle="round,pad=0.05",
                          facecolor=colors['ensemble'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box4)
    ax.text(7, 3.1, '10 Ensemble Prediction Files', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(7, 2.65, 'meta_vote | enhanced_3agree | simcse_bert | mega_ensemble | top3 | prob_avg | ...',
            fontsize=8, ha='center', va='center', style='italic')

    # Arrows from models to ensemble
    for name, x in model_names:
        ax.annotate('', xy=(x + 1.1, 3.8), xytext=(x + 1.1, 4.4),
                    arrowprops=dict(arrowstyle='->', color=colors['border'], lw=1.5))

    # Stage 5: Final Voting
    box5 = FancyBboxPatch((4.5, 0.3), 5, 1.5, boxstyle="round,pad=0.05",
                          facecolor=colors['output'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box5)
    ax.text(7, 1.2, 'Hard Voting (7/10 Agree)', fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(7, 0.75, 'Final: 479 Positives (58.85%)', fontsize=9, ha='center', va='center', style='italic')

    # Arrow ensemble to final
    ax.annotate('', xy=(7, 1.85), xytext=(7, 2.15),
                arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))

    # Score badge
    score_box = FancyBboxPatch((10.5, 0.5), 3, 1.1, boxstyle="round,pad=0.1",
                               facecolor='#4CAF50', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(score_box)
    ax.text(12, 1.05, 'F1 Score: 0.4508', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')

    plt.tight_layout()
    plt.savefig('assets/pipeline_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/pipeline_architecture.png")


def create_bert_architecture():
    """Create BERT model architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    colors = {
        'input': '#BBDEFB',
        'bert': '#C8E6C9',
        'pooler': '#FFE0B2',
        'classifier': '#F8BBD9',
        'output': '#E1BEE7',
        'border': '#37474F'
    }

    # Title
    ax.text(6, 7.5, 'BERT-based Classification Architecture',
            fontsize=14, fontweight='bold', ha='center', va='center', color='#1565C0')

    # Input Layer
    box1 = FancyBboxPatch((1, 6), 10, 0.8, boxstyle="round,pad=0.03",
                          facecolor=colors['input'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box1)
    ax.text(6, 6.4, 'Input: [CLS] + Text Tokens + [SEP]', fontsize=10, fontweight='bold', ha='center')

    # BERT Encoder
    box2 = FancyBboxPatch((1, 4), 10, 1.5, boxstyle="round,pad=0.03",
                          facecolor=colors['bert'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box2)
    ax.text(6, 5, 'klue/bert-base (Korean BERT)', fontsize=11, fontweight='bold', ha='center')
    ax.text(6, 4.5, '12 Transformer Layers | 768 Hidden | 12 Attention Heads',
            fontsize=9, ha='center', style='italic')

    # Arrow
    ax.annotate('', xy=(6, 5.55), xytext=(6, 5.95),
                arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))

    # Pooler
    box3 = FancyBboxPatch((3, 2.5), 6, 1, boxstyle="round,pad=0.03",
                          facecolor=colors['pooler'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box3)
    ax.text(6, 3, '[CLS] Token Pooling + Dropout (0.1)', fontsize=10, fontweight='bold', ha='center')

    # Arrow
    ax.annotate('', xy=(6, 3.55), xytext=(6, 3.95),
                arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))

    # Classifier
    box4 = FancyBboxPatch((3.5, 1.2), 5, 0.9, boxstyle="round,pad=0.03",
                          facecolor=colors['classifier'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box4)
    ax.text(6, 1.65, 'Linear (768 → 2) + Softmax', fontsize=10, fontweight='bold', ha='center')

    # Arrow
    ax.annotate('', xy=(6, 2.15), xytext=(6, 2.45),
                arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))

    # Output
    box5 = FancyBboxPatch((4, 0.2), 4, 0.7, boxstyle="round,pad=0.03",
                          facecolor=colors['output'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box5)
    ax.text(6, 0.55, 'Output: P(completed)', fontsize=10, fontweight='bold', ha='center')

    # Arrow
    ax.annotate('', xy=(6, 0.95), xytext=(6, 1.15),
                arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))

    plt.tight_layout()
    plt.savefig('assets/bert_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/bert_architecture.png")


def create_ensemble_voting():
    """Create ensemble voting visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'Ensemble Hard Voting Strategy (7/10 Agreement)',
            fontsize=14, fontweight='bold', ha='center', va='center', color='#1565C0')

    # 10 model boxes
    models = [
        ('meta_vote_both', 470, '#E3F2FD'),
        ('enhanced_3agree', 617, '#FFF3E0'),
        ('simcse_bert_4agree', 491, '#E8F5E9'),
        ('mega_ensemble_3agree', 483, '#FCE4EC'),
        ('top3_2agree', 483, '#F3E5F5'),
        ('prob_avg_035', 468, '#E0F7FA'),
        ('10models_7agree', 511, '#FBE9E7'),
        ('5models_4agree', 476, '#F1F8E9'),
        ('bert_data', 676, '#EDE7F6'),
        ('5models_3agree', 617, '#FFF8E1'),
    ]

    for i, (name, pos, color) in enumerate(models):
        row = i // 5
        col = i % 5
        x = 0.5 + col * 2.7
        y = 5.5 - row * 2

        box = FancyBboxPatch((x, y), 2.4, 1.5, boxstyle="round,pad=0.03",
                             facecolor=color, edgecolor='#37474F', linewidth=1.5)
        ax.add_patch(box)

        # Model name (shortened)
        short_name = name.replace('submission_', '').replace('.csv', '')
        if len(short_name) > 12:
            short_name = short_name[:11] + '...'
        ax.text(x + 1.2, y + 1.0, short_name, fontsize=8, fontweight='bold', ha='center')
        ax.text(x + 1.2, y + 0.5, f'{pos} pos', fontsize=8, ha='center', style='italic')

    # Voting aggregation
    box_vote = FancyBboxPatch((4, 1), 6, 1.2, boxstyle="round,pad=0.05",
                              facecolor='#FFECB3', edgecolor='#37474F', linewidth=2)
    ax.add_patch(box_vote)
    ax.text(7, 1.75, 'Σ votes ≥ 7 → Positive (1)', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 1.3, 'Hard Voting Threshold', fontsize=9, ha='center', style='italic')

    # Arrows from models to voting
    for i in range(10):
        row = i // 5
        col = i % 5
        x = 0.5 + col * 2.7 + 1.2
        y = 5.5 - row * 2
        ax.annotate('', xy=(7, 2.25), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='#78909C', lw=1, alpha=0.6))

    # Final result
    result_box = FancyBboxPatch((10.5, 1), 3, 1.2, boxstyle="round,pad=0.05",
                                facecolor='#4CAF50', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(result_box)
    ax.text(12, 1.75, '479 Positives', fontsize=11, fontweight='bold', ha='center', color='white')
    ax.text(12, 1.3, '58.85%', fontsize=10, ha='center', color='white')

    # Arrow
    ax.annotate('', xy=(10.4, 1.6), xytext=(10.1, 1.6),
                arrowprops=dict(arrowstyle='->', color='#37474F', lw=2))

    plt.tight_layout()
    plt.savefig('assets/ensemble_voting.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/ensemble_voting.png")


if __name__ == '__main__':
    create_pipeline_diagram()
    create_bert_architecture()
    create_ensemble_voting()
    print("\nAll diagrams generated successfully!")
