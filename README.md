# Benchmarking Model 1: Protein Sequence-Ligand Binding Prediction

This project benchmarks deep learning models (ESM-2 based) across multiple protein-ligand datasets. It includes preprocessing, embedding generation, model training, evaluation, and results tracking.

--------------------------------------------------------------------------------

Project Structure:

benchmarking_model1/
├── Dataset/               # All cleaned & formatted .pkl datasets
│   ├── UniProtSMB/
│   ├── ProteinNet/
│   ├── FLIP/
│   └── BindingDB/
│
├── Results/
│   ├── logs/              # TensorBoard logs
│   │   ├── ProteinNet/
│   │   └── FLIP/
│   └── metrics/           # Evaluation metrics (.csv)
│
├── Models/                # Final trained model checkpoints
│   ├── saved_model_FLIP.ckpt
│   └── saved_model_ProteinNet.ckpt
│
├── scripts/               # Training and preprocessing scripts
│   ├── generate_embeddings_flip.py
│   ├── triplet_flip.py
│   └── preprocessing/
│       ├── preprocess_doublesub.py
│       ├── preprocess_doublesub_fixed.py
│       └── split_flip_doublesub.py
│
├── utils/                 # Utility modules
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── inference.py
│   ├── count.py
│   └── pre.py
│
├── triplet_classification/ # Legacy & ProteinNet triplet model scripts
│   ├── triplet.py
│   ├── triplet_optimized.py
│   └── updated_triplet_model_v2.py
│
├── environment.yml        # Conda environment file
└── README.md

--------------------------------------------------------------------------------

Features:
- Preprocessing pipelines for FLIP and ProteinNet datasets
- Embedding generation using facebook/esm2_t33_650M_UR50D
- Triplet loss–based training for representation learning
- Evaluation metrics and TensorBoard support
- Model checkpoint saving and result logging

--------------------------------------------------------------------------------

How to Run:

1. Set up the environment:
   conda env create -f environment.yml
   conda activate benchmarking

2. Preprocess a dataset:
   python scripts/preprocessing/preprocess_doublesub.py

3. Generate embeddings:
   python utils/generate_all_embeddings.py --dataset FLIP

4. Train a model:
   python scripts/triplet_flip.py

5. Evaluate:
   python scripts/evaluate_all.py --dataset FLIP

--------------------------------------------------------------------------------

Results:

Final evaluation metrics are stored in Results/metrics/
TensorBoard logs are stored in Results/logs/

--------------------------------------------------------------------------------

Acknowledgments:

This project uses the facebook/esm pretrained transformer models for protein sequence representation:
https://github.com/facebookresearch/esm

--------------------------------------------------------------------------------

Author:
Maintained by Anagha Siddhi
https://github.com/anaghasiddhi
