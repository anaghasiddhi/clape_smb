
import pandas as pd
import os

# Paths
input_path = "../datasets/BindingDB/BindingDB_All.tsv"
output_dir = "../datasets/BindingDB/parsed_data"
os.makedirs(output_dir, exist_ok=True)

# Try to load only first 10K rows for dev; remove nrows=... when scaling
print("🔍 Reading TSV file...")
df = pd.read_csv(input_path, sep="\t", low_memory=False)

# Inspect column options
available_columns = df.columns.tolist()

# Candidate columns
smiles_col = "Ligand SMILES"
sequence_col = "BindingDB Target Chain Sequence"
label_col_candidates = ["Ki (nM)", "IC50 (nM)", "Kd (nM)"]

# Pick first available label column
label_col = next((col for col in label_col_candidates if col in available_columns), None)

if not all(col in available_columns for col in [smiles_col, sequence_col]) or label_col is None:
    raise ValueError("❌ Required columns not found in BindingDB TSV.")

# Drop rows with missing data
df = df[[smiles_col, sequence_col, label_col]].dropna()

# Optional: Convert label to float
df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
df = df.dropna(subset=[label_col])

# Create output in format: ID<TAB>SEQUENCE<TAB>LABEL
print("🧪 Writing parsed dataset...")
with open(os.path.join(output_dir, "bindingdb_clean.txt"), "w") as f:
    for i, row in df.iterrows():
        smile = row[smiles_col]
        seq = row[sequence_col]
        label = row[label_col]
        f.write(f"{i}\t{seq}\t{label}\n")

print(f"✅ Done! Total entries: {len(df)} written to bindingdb_clean.txt")
