# GPU-Accelerated Protein Structure Prediction

## What is Protein Structure Prediction?

Proteins are the molecular machines that power all living things. They're made up of long chains of amino acids (represented by letters like A, C, G, T), but to function properly, these chains must fold into precise 3D shapes. Understanding these shapes is crucial for:

- **Drug Discovery**: Designing medications that interact with specific proteins
- **Disease Research**: Understanding how protein misfolding causes diseases like Alzheimer's
- **Biotechnology**: Engineering proteins for industrial and medical applications

For decades, determining protein structures required expensive lab experiments taking months or years. Today, AI models can predict these structures in minutes with remarkable accuracy—but they require enormous computational power. This is where **GPU acceleration** becomes essential.

## Demo Overview

This demonstration showcases two cutting-edge AI models for protein structure prediction:

### 🧬 Boltz-2
An advanced deep learning model that predicts protein structures with high accuracy. Boltz-2 can handle:
- Single protein chains
- Multi-chain protein complexes
- Protein-ligand interactions

### 🧬 Chai-1
A state-of-the-art model optimized for complex biomolecular structures, including:
- Antibody-antigen interactions (crucial for vaccine development)
- Protein-protein complexes
- Multi-domain proteins

Both models utilize **GPU acceleration** to deliver predictions in minutes rather than hours.

## What's Included

- **Interactive Jupyter Notebooks**: User-friendly interfaces to run predictions
- **Pre-configured Examples**: 
  - Insulin structure (a well-studied hormone)
  - Antibody-antigen complex (relevant for immunology research)
- **Visualization Tools**: 3D molecular viewers to explore predicted structures
- **GPU Optimization**: Automatic detection and utilization of available GPU hardware

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- 16GB+ RAM

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the notebooks:
```bash
jupyter notebook boltz.ipynb  # For Boltz-2 predictions
# or
jupyter notebook chai.ipynb   # For Chai-1 predictions
```

### Running Predictions

Each notebook provides a simple interface where you can:

1. **Input protein sequences** (or use pre-loaded examples)
2. **Click "Run"** to start the prediction
3. **View results** including:
   - 3D structure visualization
   - Confidence scores (how certain the model is)
   - Downloadable structure files for further analysis

## Understanding the Insulin Example

Insulin is one of the most important hormones in the human body—it regulates blood sugar levels. When you run the insulin example in this demo, you'll see something fascinating:

**What you're looking at:**
The predicted structure shows insulin as two separate protein chains (called A-chain and B-chain) twisted together. Think of it like two ribbons carefully intertwined:

- **The A-chain** (21 amino acids): Appears as one colored ribbon in the visualization
- **The B-chain** (30 amino acids): Appears as a different colored ribbon, wrapping around the first

**Why this matters:**
Insulin is actually produced as one long chain in your body, but it gets cut into these two pieces that stay connected by chemical bonds (disulfide bridges—like molecular staples). This specific shape is critical:

- The folded structure creates a **binding site** where insulin can attach to receptors on your cells
- If insulin doesn't fold correctly, it can't signal cells to absorb sugar from the bloodstream
- This is why **protein structure** is directly linked to function—wrong shape = wrong function = disease

**What the colors mean:**
In the 3D visualization, you'll typically see:
- **Blue to red coloring**: Often represents confidence scores (blue = very confident, red = less certain)
- **Ribbon structures**: Show the backbone path of the protein chain
- **Stick figures**: Represent the chemical bonds holding it together

When you rotate the 3D model, notice how compact and precise the structure is—this isn't random! Evolution has optimized this exact shape over millions of years. Our GPU-powered AI models can now predict this shape from just the sequence of letters (amino acids) in minutes.

## Learn More

- **AlphaFold**: The breakthrough that revolutionized the field ([Nature paper](https://www.nature.com/articles/s41586-021-03819-2))
