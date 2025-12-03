"""IGV visualization utilities for Enformer predictions."""

import igv_notebook
from IPython.display import Markdown, display

TRACK_COLORS = {
    "DNASE": "#e63946",
    "H3K27ac": "#2a9d8f",
    "H3K4me3": "#e9c46a",
    "CAGE": "#264653",
}


def create_browser(locus, track_files, genome="hg38"):
    """Create IGV browser with Enformer prediction tracks."""
    igv_notebook.init()
    
    tracks = [{
        "name": "RefSeq Genes",
        "type": "annotation",
        "format": "refgene",
        "url": "https://s3.amazonaws.com/igv.org.genomes/hg38/refGene.txt.gz",
        "indexed": False,
        "visibilityWindow": 300000,
        "displayMode": "EXPANDED"
    }]
    
    for name, path in track_files:
        color = next((c for k, c in TRACK_COLORS.items() if k in name), "#666")
        tracks.append({
            "name": f"Enformer: {name}",
            "type": "wig",
            "format": "bedGraph",
            "path": path,
            "color": color,
            "height": 60,
            "autoscale": True
        })
    
    return igv_notebook.Browser({
        "genome": genome,
        "locus": locus,
        "tracks": tracks
    })


def show_interpretation_guide():
    """Display interpretation guide for Enformer tracks."""
    display(Markdown("""
---
### Interpreting These Tracks

**CAGE:K562** — Marks transcription start sites (TSSs). Peaks indicate where RNA polymerase begins transcribing genes. Sharp peaks at gene starts confirm active transcription in K562 cells.

**DNASE** — Shows open chromatin regions. High signal = accessible DNA where regulatory proteins can bind. Compare K562 vs GM12878 to see cell-type-specific regulation.

**H3K27ac** — Histone mark for active enhancers and promoters. Strong signal near genes suggests active regulatory elements driving expression.

**H3K4me3** — Histone mark enriched at active promoters. Peaks typically sit right at TSSs of actively transcribed genes.

**Why this matters**: Enformer predicts these signals directly from DNA sequence, enabling variant effect prediction, gene regulation analysis, and understanding how mutations might disrupt regulatory elements—all without running wet lab experiments.
"""))