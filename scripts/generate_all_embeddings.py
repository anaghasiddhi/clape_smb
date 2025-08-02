import sys
import os
import torch
import esm
import pickle
from tqdm import tqdm

if len(sys.argv) != 2:
    print("Usage: python generate_embeddings.py <DATASET_NAME>")
    sys.exit(1)

dataset = sys.argv[1]  # e.g., FLIP, ProteinNet, etc.

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Ensure output directory
os.makedirs("./Dataset/{dataset}", exist_ok=True)

# Process each split
splits = ["train", "valid", "test"]
for split in splits:
    print(f"\nProcessing {split} split...")
    input_path = f"../Dataset/{dataset}/splits/{split}_flip.txt"
    output_path = f"../Dataset/{dataset}/esm_{split}_FLIP.pkl"

    data_dict = {}

    with open(input_path, "r") as f:
        for line in tqdm(f, desc=f"{split} progress"):
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            pid, sequence, label_str = parts
            label = [int(c) for c in label_str.strip()]

            if len(sequence) != len(label):
                print(f"⚠️Skipping {pid}: sequence and label lengths don't match.")
                continue

            seq = [(pid, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(seq)

            with torch.no_grad():
                output = model(batch_tokens, repr_layers=[6])

            embedding = output["representations"][6].squeeze(0)[1:len(sequence)+1]
            data_dict[pid] = (embedding.cpu(), torch.tensor(label, dtype=torch.long))

    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f" Saved embeddings to {output_path}")
