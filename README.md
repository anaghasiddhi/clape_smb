
# CLAPE-SMB: Transformer-Based Protein Sequence–Ligand Binding Site Prediction

This repository contains the codebase for **CLAPE-SMB**, a deep learning model built on top of ESM-2 protein language models for predicting ligand binding sites from protein sequences. It was developed as part of a Master's thesis in computational biology and machine learning.

CLAPE-SMB uses triplet loss–based training for fine-grained residue-level supervision, and is benchmarked on the UniProtSMB and BioLiP datasets.

---
## Project Structure

```text
clape-smb/
├── Dataset/                    # (excluded) Placeholder for UniProtSMB and BioLiP
├── evaluate/                   # Evaluation scripts and TensorBoard logs
│   └── tensorboard_logs/
├── scripts/                    # Embedding generation and preprocessing
│   └── generate_all_embeddings_distributed.py
├── utils/                      # Model, loss functions, inference utilities
│   ├── model.py
│   ├── losses.py
│   ├── inference.py
│   └── ...
├── Results/                    # Metrics and model checkpoints (not tracked by Git)
├── triplet_main.py             # Main training script
├── environment.yml             # Conda environment file
└── README.md
```

---

## Features

- Embedding generation using ESM2 (facebook/esm2_t33_650M_UR50D)
- Transformer-based sequence model with triplet loss training
- Residue-level prediction and evaluation (Precision, Recall, F1, AUROC)
- Evaluation logging with TensorBoard
- Modular structure for training, evaluation, and inference

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate clape_smb

```
How to Run
Note: Dataset contents are excluded from version control. See Dataset/README_dataset_structure.md for details.

1) Generate ESM-2 embeddings:
```
python scripts/generate_all_embeddings_distributed.py UniProtSMB
```
2) Train the model:
```
python triplet_main.py
```
3) Run evaluation:
```
python evaluate/evaluate_model.py
```
4) Launch TensorBoard to view logs:
```
tensorboard --logdir evaluate/tensorboard_logs/
```

## Dataset Format
The following subdirectories must exist under Dataset/:

BioLiP/: Contains processed structural binding site labels and metadata

UniProtSMB/: Contains protein sequences and precomputed embeddings

Refer to Dataset/README_dataset_structure.md for expected formats and file structures.

## Results
Evaluation metrics are stored in Results/metrics/

Model checkpoints are stored in Results/logs/**/checkpoints/

Inference outputs and training logs are available under evaluate/tensorboard_logs/

---

## Citation

If you use this repository or derive from the CLAPE-SMB model, please cite the following works:

**CLAPE-SMB (2025)**  
Transformer-based protein sequence–ligand binding site prediction model.  
Developed as part of a Master's thesis at Oklahoma State University.

### Benchmarking Methodology

Wang, Jue, Liu, Yufan, and Tian, Boxue.  
**Protein-small molecule binding site prediction based on a pre-trained protein language model with contrastive learning.**  
*Journal of Cheminformatics*, 16(1), 125 (2024).  
[https://doi.org/10.1186/s13321-024-00920-2](https://doi.org/10.1186/s13321-024-00920-2)

### ESM-2 Protein Language Model

Lin, Zeming, Akin, Halil, Rao, Roshan, Hie, Brian, Zhu, Zhongkai, Lu, Wenting, Smetanin, Nikita, dos Santos Costa, Allan, Fazel-Zarandi, Maryam, Sercu, Tom, Candido, Sal, *et al.*  
**Language models of protein sequences at the scale of evolution enable accurate structure prediction.**  
*bioRxiv* (2022).  
[https://doi.org/10.1101/2022.07.20.500902](https://doi.org/10.1101/2022.07.20.500902)

### BioLiP Structural Binding Annotations

Yang J, Roy A, Zhang Y.  
**BioLiP: a semi-manually curated database for biologically relevant ligand-protein interactions.**  
*Nucleic Acids Research*, 41(Database issue): D1096–D1103 (2013).  
[https://doi.org/10.1093/nar/gks966](https://doi.org/10.1093/nar/gks966)

---

## Acknowledgments

- **CLAPE-SMB** builds on pretrained protein embeddings from [facebookresearch/esm](https://github.com/facebookresearch/esm) (ESM-2).
- Binding site annotations are curated from the [BioLiP database](https://zhanggroup.org/BioLiP/).
- This work was conducted in the Department of Computer Science at Oklahoma State University as part of a Master's thesis in computational biology and machine learning.



--------------------------------------------------------------------------------

Author:
Maintained by Anagha Siddhi
https://github.com/anaghasiddhi
