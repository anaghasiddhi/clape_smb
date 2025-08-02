import pandas as pd
import os

# Constants
WT_SEQ = list("MQDRFSRQLNADAVRLLVHGADVNGLIYNRGDQASQHVDIEEGDTVTGTLN")
WT_LEN = len(WT_SEQ)

INPUT_FILE = "../Dataset/FLIP/DoubleSub.txt"
OUTPUT_DIR = "../Dataset/FLIP/parsed_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "flip_doublesub_clean.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the double substitution file
df = pd.read_csv(INPUT_FILE, sep="\t")

# Drop rows with missing fitness
df = df.dropna(subset=["ExpectedFit"])

skipped = 0
with open(OUTPUT_FILE, "w") as f:
    for i, row in df.iterrows():
        try:
            seq = WT_SEQ.copy()
            pos1 = int(row["Substitution1-Pos"]) - 1
            pos2 = int(row["Substitution2-Pos"]) - 1

            if pos1 >= WT_LEN or pos2 >= WT_LEN:
                skipped += 1
                continue

            seq[pos1] = row["Substitution1-Mutaa"]
            seq[pos2] = row["Substitution2-Mutaa"]
            mutated_seq = ''.join(seq)
            f.write(f"{i}\t{mutated_seq}\t{row['ExpectedFit']}\n")
        except Exception as e:
            skipped += 1

print(f"✅ Done. Cleaned file written to: {OUTPUT_FILE}")
print(f"⚠️ Total rows skipped due to errors or out-of-bounds: {skipped}")
