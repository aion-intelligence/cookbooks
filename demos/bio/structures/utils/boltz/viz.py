
import matplotlib.pyplot as plt
import numpy as np


def plot_confidence_scores(confidence_data: dict) -> None:
    """Plot Boltz-2 confidence metrics"""
    
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics = {
        'Confidence\nScore': confidence_data.get('confidence_score', 0),
        'pTM': confidence_data.get('ptm', 0),
        'ipTM': confidence_data.get('iptm', 0),
        'Complex\npLDDT': confidence_data.get('complex_plddt', 0),
        'Complex\nipLDDT': confidence_data.get('complex_iplddt', 0),
    }

    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#2E86AB' if v > 0.8 else '#A23B72' if v > 0.6 else '#F18F01' for v in values]

    bars = axes[0].bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Overall Confidence Metrics', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High confidence (>0.8)')
    axes[0].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Medium confidence (>0.6)')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    chains_ptm = confidence_data.get('chains_ptm', {})
    pair_chains_iptm = confidence_data.get('pair_chains_iptm', {})

    if chains_ptm and pair_chains_iptm:
        chain_ids = sorted(chains_ptm.keys())
        n_chains = len(chain_ids)
        interaction_matrix = np.zeros((n_chains, n_chains))

        for i, chain_i in enumerate(chain_ids):
            for j, chain_j in enumerate(chain_ids):
                if chain_i in pair_chains_iptm and chain_j in pair_chains_iptm[chain_i]:
                    interaction_matrix[i, j] = pair_chains_iptm[chain_i][chain_j]

        im = axes[1].imshow(interaction_matrix, cmap='RdYlGn', vmin=0, vmax=1.0, aspect='auto')
        axes[1].set_xticks(range(n_chains))
        axes[1].set_yticks(range(n_chains))
        axes[1].set_xticklabels([f'Chain {c}' for c in chain_ids])
        axes[1].set_yticklabels([f'Chain {c}' for c in chain_ids])
        axes[1].set_title('Chain Interaction Confidence (ipTM)', fontsize=14, fontweight='bold')

        for i in range(n_chains):
            for j in range(n_chains):
                text = axes[1].text(j, i, f'{interaction_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label('ipTM Score', fontsize=11)
    else:
        axes[1].text(0.5, 0.5, 'No chain interaction data available',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Chain Interaction Confidence', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()