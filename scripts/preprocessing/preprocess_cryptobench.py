import os
import json
import random
import requests
import time

# === CONFIG ===
RAW_JSON = "Raw_data/Cryptobench/dataset.json"
CACHE_FILE = "Raw_data/Cryptobench/uniprot_sequences.json"
OUT_DIR = "Dataset/Cryptobench/splits"
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SEED = 42
RETRY_DELAY = 2  # seconds between failed requests

# === UniProt Fetch ===
def fetch_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {uniprot_id} (status {response.status_code})")
            return None
        lines = response.text.strip().split("\n")
        return ''.join(lines[1:])  # skip FASTA header
    except Exception as e:
        print(f"Error fetching {uniprot_id}: {e}")
        return None

def get_sequence(uniprot_id, cache):
    if uniprot_id in cache:
        return cache[uniprot_id]
    
    sequence = fetch_sequence(uniprot_id)
    if sequence:
        cache[uniprot_id] = sequence
        # Save after every successful fetch
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    else:
        time.sleep(RETRY_DELAY)
    return sequence

def make_label(seq, pocket_positions):
    label = ['0'] * len(seq)
    for pos in pocket_positions:
        try:
            _, index = pos.split("_")
            idx = int(index) - 1
            if 0 <= idx < len(seq):
                label[idx] = '1'
        except:
            continue
    return ''.join(label)

# === Load data ===
with open(RAW_JSON, "r") as f:
    data = json.load(f)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

entries = []

for holo_list in data.values():
    for entry in holo_list:
        uid = entry.get("uniprot_id")
        if not uid:
            continue
        seq = get_sequence(uid, cache)
        if not seq:
            continue
        label = make_label(seq, entry.get("apo_pocket_selection", []))
        if len(seq) != len(label):
            print(f"Skipping {uid}: sequence and label length mismatch ({len(seq)} vs {len(label)})")
            continue
        entries.append((uid, seq, label))

print(f"Total usable entries: {len(entries)}")

# === Shuffle & split ===
random.seed(SEED)
random.shuffle(entries)

n = len(entries)
n_train = int(n * SPLIT_RATIOS[0])
n_valid = int(n * SPLIT_RATIOS[1])

splits = {
    "train": entries[:n_train],
    "valid": entries[n_train:n_train + n_valid],
    "test": entries[n_train + n_valid:]
}

# === Save ===
os.makedirs(OUT_DIR, exist_ok=True)

for split_name, split_data in splits.items():
    out_path = os.path.join(OUT_DIR, f"{split_name}_cryptobench.txt")
    with open(out_path, "w") as f:
        for pid, seq, label in split_data:
            f.write(f"{pid}\t{seq}\t{label}\n")
    print(f"Saved {len(split_data)} entries to {out_path}")
