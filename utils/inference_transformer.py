import torch
import argparse
import pickle
from tqdm import tqdm
from model import TransformerModel
import esm
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter(log_dir="runs/clape_inference_probs")

# ----------------------- Args -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Input TSV: ID \\t SEQUENCE \\t LABEL (label ignored)')
parser.add_argument('--output', required=True, help='Output path for predictions')
parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
parser.add_argument('--model_path', required=True, help='Path to .pt file')
parser.add_argument('--use_cached_embeddings', action='store_true',
                    help='Use precomputed ESM2 embeddings')
parser.add_argument('--embedding_dir', type=str, default=None,
                    help='Directory containing cached esm_{split}_BioLiP_rank*_part*.pkl files')
args = parser.parse_args()

# ----------------------- Device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------- Load ESM2 -----------------------
if not args.use_cached_embeddings:
    print("Loading ESM2 model...")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    esm_model = esm_model.to(device)
else:
    esm_model = None
    batch_converter = None

# ----------------------- Load Classifier -----------------------
print("Loading trained TransformerModel...")
model = TransformerModel()
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
model = model.to(device)

# Load projection layer
project = torch.nn.Linear(1280, 320).to(device)

# ----------------------- Load Cached Embeddings -----------------------
cached_embeddings = {}
if args.use_cached_embeddings:
    print("🔄 Loading cached ESM2 embeddings...")
    for pkl_path in glob.glob(os.path.join(args.embedding_dir, "*.pkl")):
        if pkl_path.endswith(".tmp"):
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)  # {pid: embedding}
            cached_embeddings.update(data)
    print(f"✅ Loaded embeddings for {len(cached_embeddings)} proteins.")

# ----------------------- Read TSV -----------------------
print("Reading input...")
seq_dict = {}
with open(args.input, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        pid, seq = parts[0], parts[1]
        seq_dict[pid] = seq.upper()

# ----------------------- Inference -----------------------
print("Running inference...")
results = []

for pid, seq in tqdm(seq_dict.items()):
    if args.use_cached_embeddings:
        if pid not in cached_embeddings:
            print(f"[!] Missing cached embedding for {pid}, skipping.")
            continue
        rep_data = cached_embeddings[pid]
        if isinstance(rep_data, tuple):
            rep_np = rep_data[0]
        else:
            rep_np = rep_data

        if not isinstance(rep_np, np.ndarray) or rep_np.ndim != 2:
            print(f"[!] Bad embedding format for {pid}, skipping.")
            continue

        rep = torch.from_numpy(rep_np).float().unsqueeze(0).to(device)

    else:
        batch_labels, batch_strs, batch_tokens = batch_converter([(pid, seq)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            rep = esm_model(batch_tokens, repr_layers=[33])["representations"][33]
        rep = rep.squeeze(0)[1:-1, :].unsqueeze(0).to(device)  # (1, L, 1280)

    rep = project(rep)  # (1, L, 320)

    with torch.no_grad():
        scores, _ = model(rep)
        probs = torch.softmax(scores.squeeze(0), dim=-1)[:, 1]  # (L,)
        pred = ''.join(['1' if p > args.threshold else '0' for p in probs])

        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            count = (probs > t).sum().item()
            writer.add_scalar(f"{pid}/count_above_{t}", count, 0)

    for i, p in enumerate(probs):
        writer.add_scalar(f"{pid}/prob_residue", p.item(), i)

    writer.add_histogram(f"{pid}/distribution", probs, 0)

    probs_str = ' '.join([f"{p:.4f}" for p in probs.tolist()])
    results.append((pid, seq, pred, probs_str))

# ----------------------- Write Output -----------------------
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, 'w') as f:
    for pid, seq, pred, probs_str in results:
        f.write(f"{pid}\n{seq}\n{pred}\n{probs_str}\n")

print(f"✅ Done! Predictions saved to: {args.output}")

writer.close()
