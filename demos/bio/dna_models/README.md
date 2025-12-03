# Gene Regulation Prediction with Enformer (DeepMind)

## What is Gene Regulation?

Your DNA contains the instructions to make thousands of proteins, but cells don't use all these instructions at once. **Gene regulation** is the process that controls which genes are turned "on" or "off" in different cell types and conditions. This is why a heart cell looks and acts differently from a brain cell, even though they have the exact same DNA.

Understanding gene regulation is critical for:

- **Disease Research**: Many diseases result from genes being turned on/off at the wrong time or place
- **Personalized Medicine**: Predicting how genetic variants affect gene activity in different individuals
- **Drug Development**: Identifying which genes drugs affect and in which tissues
- **Cancer Biology**: Understanding how cancer cells hijack normal gene regulation

Traditionally, studying gene regulation required expensive lab experiments (like ChIP-seq, ATAC-seq, CAGE) that can cost thousands of dollars per sample and take weeks to complete. **Enformer changes this** by predicting regulatory activity directly from DNA sequence using AI.

## What is Enformer?

Enformer is a state-of-the-art deep learning model developed by DeepMind (the team behind AlphaFold) that predicts gene regulatory activity from raw DNA sequence. It's trained on thousands of experimental datasets and can predict:

- **Chromatin accessibility** (where DNA is "open" for regulatory proteins to bind)
- **Histone modifications** (chemical markers that indicate active or silent genes)
- **Transcription start sites** (where genes begin being transcribed into RNA)
- **Regulatory element activity** (enhancers, promoters, silencers)

**Key capability**: Given a DNA sequence, Enformer predicts what the regulatory landscape would look like across different cell types—without running any experiments.

## Why GPUs Are Essential

Enformer is a transformer-based neural network with over 100 million parameters. Processing a single DNA sequence requires:
- Analyzing ~400,000 base pairs simultaneously
- Computing billions of attention weights across the sequence
- Generating predictions for 5,313 different regulatory tracks

**GPU acceleration makes this practical:**
- **Predictions in seconds** instead of minutes on CPU
- **Interactive exploration** of different genomic regions
- **Batch processing** for analyzing multiple variants or regions
- **Real-time variant effect prediction** for clinical applications

Our GPU infrastructure enables researchers to run these complex models efficiently at scale.

## Demo Overview: Exploring the TP53 Gene

This notebook demonstrates Enformer by analyzing the **TP53 gene** region—one of the most important cancer-related genes in the human genome. TP53 is often called the "guardian of the genome" because it helps prevent cancer by controlling cell division.

### What the Demo Shows

When you run the notebook, you'll see:

1. **CAGE Tracks** (Transcription Start Sites)
   - Sharp peaks showing where the TP53 gene starts being transcribed
   - Different heights in different cell types reflect tissue-specific expression

2. **DNASE Tracks** (Open Chromatin)
   - Shows which parts of DNA are accessible to regulatory proteins
   - Compare K562 (leukemia cells) vs GM12878 (normal B-cells) to see how the same DNA is used differently
   - High signal = "open for business" where transcription factors can bind

3. **Histone Modification Tracks**
   - **H3K27ac**: Marks active enhancers and promoters (like "on" switches for genes)
   - **H3K4me3**: Concentrated at active gene promoters (marks the start of active genes)
   - These are chemical modifications to proteins that DNA wraps around

### What You're Looking At

The visualization shows a ~115kb region around TP53. Each horizontal track represents a different type of regulatory signal:

- **Peaks and valleys**: High peaks = strong regulatory activity at that location
- **Different colors**: Each track type has its own color for easy identification
- **Multiple cell types**: Same DNA, different regulatory patterns based on cell type
- **Gene annotations**: Reference genes shown at the bottom for context

**The remarkable part**: All these predictions come from just the DNA sequence. Enformer learned to predict chromatin state and gene activity without ever "seeing" the actual experimental data for this specific region.

## Real-World Applications

### Variant Effect Prediction
Predict how genetic variants (mutations, SNPs) affect gene regulation. For example:
- Will this variant disrupt a transcription factor binding site?
- Does this mutation affect TP53 expression levels?
- How does this variant's effect differ between cell types?

### Disease Research
Identify regulatory variants associated with diseases:
- Non-coding variants that increase cancer risk
- Mutations in enhancers that cause developmental disorders
- Understanding why some people respond differently to drugs

### Drug Target Discovery
Predict which genes are affected by compounds in different tissues, helping prioritize drug targets and anticipate side effects.

### Synthetic Biology
Design synthetic regulatory elements (promoters, enhancers) with predicted activity profiles for biotechnology applications.

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x with GPU support
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- ~2GB disk space for reference genome

### Running the Demo

1. Open the notebook:
```bash
jupyter notebook enformer.ipynb
```

2. The notebook will:
   - Download the human reference genome (hg38) if needed
   - Load the Enformer model from TensorFlow Hub
   - Run predictions for the TP53 gene region
   - Display results in an interactive genome browser

3. Explore the results:
   - Scroll through tracks to see different regulatory signals
   - Zoom in/out to see details or broader context
   - Compare predictions across cell types

### Customization

You can modify the notebook to analyze different regions by changing:
```python
CHROM = 'chr17'      # Chromosome
CENTER = 7_580_000   # Center position (TP53 gene)
```

Try other interesting genes:
- `chr7:5,529,000` — ACTB (housekeeping gene, active everywhere)
- `chr11:5,248,000` — HBB (hemoglobin, blood-cell specific)
- `chr19:50,087,000` — APOE (Alzheimer's risk gene)

## Technical Details

**Model Architecture:**
- Transformer-based deep learning model
- Input: 393,216 bp DNA sequence (one-hot encoded)
- Output: 896 bins × 5,313 tracks = 4.76M predictions per sequence
- Output resolution: 128 bp per bin

**Performance:**
- Inference time: ~2-5 seconds per sequence on GPU (vs ~60+ seconds on CPU)
- Model size: ~250MB compressed
- Memory requirement: ~8GB GPU RAM for inference

**Prediction Tracks:**
- 5,313 total tracks covering different assays and cell types
- Includes CAGE, DNase-seq, ChIP-seq for various histone marks and transcription factors
- 896 bins covering ~115kb output region (input center ±57kb)

## Understanding the Science

### Why These Signals Matter

**CAGE (Cap Analysis of Gene Expression)**
- Identifies exact positions where genes start being transcribed
- Sharp peaks = transcription start sites (TSSs)
- Peak height correlates with gene expression level

**DNASE (DNase I Hypersensitivity)**
- Marks regions where DNA is accessible (not tightly packed)
- Open chromatin = potential regulatory activity
- Cell-type specific patterns reveal tissue-specific regulation

**H3K27ac (Histone H3 Lysine 27 Acetylation)**
- Chemical mark on histones indicating active regulatory regions
- Strong at active enhancers and promoters
- Correlates with gene activation

**H3K4me3 (Histone H3 Lysine 4 Trimethylation)**
- Enriched specifically at active promoters
- Sharp peaks at transcription start sites
- Indicates genes poised for or actively being transcribed

### The Power of Sequence-Based Prediction

Enformer demonstrates that much of gene regulation is **encoded in the DNA sequence itself**. By learning from thousands of experiments, the model captures:

- Transcription factor binding motifs
- DNA shape and accessibility preferences
- Long-range regulatory interactions (up to 100kb away)
- Cell-type-specific regulatory logic

This means we can now predict regulatory effects of genetic variants, design synthetic regulatory elements, and understand disease mechanisms—all from sequence alone, without running expensive experiments.

## Learn More

- **Original Paper**: [Effective gene expression prediction from sequence by integrating long-range interactions](https://www.nature.com/articles/s41592-021-01252-x) (Nature Methods, 2021)
- **Model on TensorFlow Hub**: [DeepMind Enformer](https://tfhub.dev/deepmind/enformer/1)
- **ENCODE Project**: [Encyclopedia of DNA Elements](https://www.encodeproject.org/) — datasets used to train Enformer
- **Roadmap Epigenomics**: [Epigenome Atlas](http://www.roadmapepigenomics.org/)

