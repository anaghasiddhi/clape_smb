import os
import sys
import torch
import esm
import pickle
from tqdm import tqdm

# Verify arguments
if len(sys.argv) != 2:
    print("Usage: torchrun --nproc_per_node=3 generate_all_embeddings.py <DATASET_NAME>")
    sys.exit(1)
dataset = sys.argv[1]

dataset_lower = dataset.lower()
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.to(device)

batch_size = 1
save_every = 10000
splits = ["train", "valid", "test"]

def safe_save_pickle(data_dict, out_path, rank, max_retries=3):
    tmp_path = out_path + ".tmp"
    for attempt in range(max_retries):
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(data_dict, f)

            # Verify saved file
            with open(tmp_path, "rb") as f:
                _ = pickle.load(f)
            os.rename(tmp_path, out_path)
            print(f"[Rank {rank}] Verified and saved: {out_path}", flush=True)
            return True
        except Exception as e:
            print(f"[Rank {rank}] Save attempt {attempt+1} failed: {out_path} ({e})", flush=True)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print(f"[Rank {rank}] Giving up on: {out_path}", flush=True)
    return False

def is_valid_pkl(path):
    try:
        with open(path, "rb") as f:
            _ = pickle.load(f)
        return True
    except:
        return False

for split in splits:
    print(f"[Rank {rank}] Processing {split}...", flush=True)

    input_path = f"./Dataset/{dataset}/splits/{split}_{dataset_lower}.txt"
    output_dir = f"./Dataset/{dataset}/intermediate_{split}"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        if rank == 0:
            print(f"File not found: {input_path}", flush=True)
        continue

    data = []
    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            pid, seq, label_str = parts
            seq = seq.upper()
            label = [int(c) for c in label_str.strip()]
            if len(seq) != len(label):
                continue
            data.append((pid, seq, label))

    shard = data[rank::world_size]
    local_results = {}
    batch_count = 0
    total_saved = 0

    for i in tqdm(range(0, len(shard), batch_size), desc=f"[Rank {rank}] {split}"):
        batch = shard[i:i + batch_size]
        batch_input = [(x[0], x[1]) for x in batch]
        batch_labels = [x[2] for x in batch]

        _, _, batch_tokens = batch_converter(batch_input)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            output = model(batch_tokens, repr_layers=[6])

        representations = output["representations"][6]
        for j, (pid, seq, label) in enumerate(batch):
            emb = representations[j, 1:len(seq)+1].cpu()
            local_results[pid] = (emb, torch.tensor(label, dtype=torch.long))
            batch_count += 1

        if batch_count >= save_every:
            out_path = os.path.join(output_dir, f"esm_{split}_{dataset}_rank{rank}_part{total_saved}.pkl")

            # Only skip if file exists and loads correctly
            if os.path.exists(out_path) and is_valid_pkl(out_path):
                print(f"[Rank {rank}] Skipping existing valid: {out_path}", flush=True)
                total_saved += 1
                batch_count = 0
                local_results.clear()
                continue

            success = safe_save_pickle(local_results, out_path, rank)
            if success:
                total_saved += 1
            local_results.clear()
            batch_count = 0

    # Final save
    if local_results:
        out_path = os.path.join(output_dir, f"esm_{split}_{dataset}_rank{rank}_part{total_saved}.pkl")

        if os.path.exists(out_path) and is_valid_pkl(out_path):
            print(f"[Rank {rank}] Skipping final existing valid: {out_path}", flush=True)
        else:
            success = safe_save_pickle(local_results, out_path, rank)
            if success:
                total_saved += 1

    print(f"[Rank {rank}] Finished all splits.", flush=True)
