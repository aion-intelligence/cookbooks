"""Enformer utilities for sequence extraction, encoding, and prediction export."""

import numpy as np
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import os
from pathlib import Path

SEQUENCE_LENGTH = 393_216
BIN_SIZE = 128
TARGET_LENGTH = 896

# Track indices from Basenji2 targets (ordered for display)
TRACK_INDICES = [
    ("CAGE:K562", 4828),
    ("DNASE:K562", 121),
    ("DNASE:GM12878", 120),
    ("H3K27ac:K562", 706),
    ("H3K4me3:K562", 715),
]

TRACK_COLORS = {
    "DNASE": "#e63946",
    "H3K27ac": "#2a9d8f",
    "H3K4me3": "#e9c46a",
    "CAGE": "#264653",
}


class FastaExtractor:
    """Extract sequences from a FASTA file with padding for edge cases."""
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chr_sizes = {k: len(v) for k, v in self.fasta.items()}
    
    def extract(self, interval):
        chr_len = self._chr_sizes[interval.chrom]
        start = max(interval.start, 0)
        end = min(interval.end, chr_len)
        seq = str(self.fasta.get_seq(interval.chrom, start + 1, end).seq).upper()
        pad_left = 'N' * max(-interval.start, 0)
        pad_right = 'N' * max(interval.end - chr_len, 0)
        return pad_left + seq + pad_right


def one_hot_encode(sequence):
    """One-hot encode DNA sequence."""
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def download_genome(fasta_path='/content/hg38.fa'):
    """Download hg38 reference genome if not present."""
    if not os.path.exists(fasta_path):
        print("Downloading hg38 reference genome (~1GB)...")
        os.system(f'wget -q -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > {fasta_path}')
        print("Done")
    else:
        print("Reference genome already exists")
    return fasta_path


def get_input_interval(chrom, center_pos):
    """Get the input interval for Enformer given a center position."""
    start = center_pos - SEQUENCE_LENGTH // 2
    end = center_pos + SEQUENCE_LENGTH // 2
    return Interval(chrom, start, end)


def get_output_region(input_start):
    """Calculate the output region coordinates from input start."""
    out_start = input_start + (SEQUENCE_LENGTH - TARGET_LENGTH * BIN_SIZE) // 2
    out_end = out_start + TARGET_LENGTH * BIN_SIZE
    return out_start, out_end


def predictions_to_bedgraph(preds, chrom, input_start, filepath):
    """Write predictions to bedGraph format."""
    out_start = input_start + (SEQUENCE_LENGTH - TARGET_LENGTH * BIN_SIZE) // 2
    with open(filepath, 'w') as f:
        for i, val in enumerate(preds):
            f.write(f"{chrom}\t{out_start + i*BIN_SIZE}\t{out_start + (i+1)*BIN_SIZE}\t{val:.4f}\n")
    return filepath


def export_tracks(predictions, chrom, input_start, output_dir):
    """Export selected tracks to bedGraph files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    track_files = []
    for name, idx in TRACK_INDICES:
        path = str(output_dir / f"{name.replace(':', '_')}.bedGraph")
        predictions_to_bedgraph(predictions[:, idx], chrom, input_start, path)
        track_files.append((name, path))
        print(f"Exported: {name}")
    
    return track_files