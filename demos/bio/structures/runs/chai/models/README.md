---
license: apache-2.0
---
# Chai-1

Chai-1 is a multi-modal foundation model for molecular structure prediction that performs at the state-of-the-art across a variety of benchmarks. Chai-1 enables unified prediction of proteins, small molecules, DNA, RNA, glycosylations, and more.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/638d7664d1c4bf949b1d2a67/25SJNV--JbPBQ8OEtCTXS.png)

For more information on the model's performance and capabilities, see our [technical report](https://www.biorxiv.org/content/10.1101/2024.10.10.615955).

## Installation

```shell
# version on pypi:
pip install chai_lab==0.4.2

# newest available version (updates daily to test features that weren't released yet):
pip install git+https://github.com/chaidiscovery/chai-lab.git
```

This Python package requires Linux, and a GPU with CUDA and bfloat16 support. We recommend using an A100 80GB or H100 80GB chip, but A10s and A30s should work for smaller complexes. Users have also reported success with consumer-grade RTX 4090.

## Running the model

The model accepts inputs in the FASTA file format, and allows you to specify the number of trunk recycles and diffusion timesteps via the `chai_lab.chai1.run_inference` function. By default, the model generates five sample predictions, and uses embeddings without MSAs or templates.

The following script demonstrates how to provide inputs to the model, and obtain a list of PDB files for downstream analysis:

```shell
python examples/predict_structure.py
```

To get the best performance, we recommend running the model with MSAs. The following script demonstrates how to provide MSAs to the model by calling out to an MSA server:

```shell
python examples/msas/predict_with_msas.py
```

## Further instructions

For further instructions, please see our instructions in the GitHub repository: https://github.com/chaidiscovery/chai-lab