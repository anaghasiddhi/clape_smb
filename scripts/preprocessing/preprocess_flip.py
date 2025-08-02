
import pandas as pd
import os

# Input & output paths
input_csv = "../datasets/FLIP/raw_repo/splits/gb1/four_mutations_full_data.csv"
output_dir = "../datasets/FLIP/parsed_data"
os.makedirs(output_dir, exist_ok=True)

# Read CSV file
print("🔍 Reading FLIP GB1 CSV...")
df = pd.read_csv(input_csv)

# Keep only sequence and fitness
df = df[["sequence", "Fitness"]].dropna()

# Convert Fitness to float
df["Fitness"] = pd.to_numeric(df["Fitness"], errors="coerce")
df = df.dropna(subset=["Fitness"])

# Save in format: ID\tSEQUENCE\tLABEL
output_file = os.path.join(output_dir, "flip_gb1_clean.txt")
print(f"📦 Writing output to {output_file}...")

with open(output_file, "w") as f:
    for i, row in df.iterrows():
        f.write(f"{i}\t{row['sequence']}\t{row['Fitness']}\n")

print(f"✅ Done! Total entries written: {len(df)}")
