import pandas as pd
import os

# Constants
WT_SEQ = list("MQDRFSRQLNADAVRLLVHGADVNGLIYNRGDQASQHVDIEEGDTVTGTLN")
INPUT_FILE = "/home/scratch1/asiddhi/benchmarking_model1/Dataset/FLIP/DoubleSub.txt"
OUTPUT_DIR = "/home/scratch1/asiddhi/benchmarking_model1/Dataset/FLIP/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "flip_doublesub_clean.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the double substitution file
df = pd.read_csv(INPUT_FILE, sep="\t")

# Drop rows with missing fitness
df = df.dropna(subset=["ExpectedFit"])

# Prepare output
with open(OUTPUT_FILE, "w") as f:
    for i, row in df.iterrows():
        seq = WT_SEQ.copy()
        try:
            pos1 = int(row["Substitution1-Pos"]) - 1
            pos2 = int(row["Substitution2-Pos"]) - 1
            seq[pos1] = row["Substitution1-Mutaa"]
            seq[pos2] = row["Substitution2-Mutaa"]
            mutated_seq = ''.join(seq)
            f.write(f"{i}\t{mutated_seq}\t{row['ExpectedFit']}\n")
        except Exception as e:
            print(f"❌ Skipping row {i} due to error: {e}")

print(f"✅ Done. Cleaned file written to: {OUTPUT_FILE}")
