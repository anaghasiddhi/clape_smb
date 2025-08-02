
import os

input_path = "../datasets/ProteinNet/raw_repo/code/text_sample"
output_dir = "../datasets/ProteinNet/parsed_data"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "proteinnet_clean.txt")

with open(input_path, "r") as infile, open(output_file, "w") as outfile:
    current_id = None
    current_seq = None
    for line in infile:
        line = line.strip()
        if line == "[ID]":
            current_id = infile.readline().strip()
        elif line == "[PRIMARY]":
            current_seq = infile.readline().strip()
        if current_id and current_seq:
            outfile.write(f"{current_id}\t{current_seq}\t0\n")
            current_id, current_seq = None, None

print(f"✅ Done! Parsed sequences written to: {output_file}")
