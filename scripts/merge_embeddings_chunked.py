import os
import pickle
import glob
import argparse

def merge_in_segments(input_dir, output_file, temp_dir, chunk_size=10):
    os.makedirs(temp_dir, exist_ok=True)
    all_chunk_files = []
    files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    print(f"Processing {len(files)} files from: {input_dir}")

    chunk_idx = 0
    current_chunk = []

    for i, f in enumerate(files):
        print(f"Loading: {f}")
        with open(f, "rb") as pf:
            data = pickle.load(pf)
            current_chunk.extend(data.items())

        if (i + 1) % chunk_size == 0 or i == len(files) - 1:
            temp_chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.pkl")
            with open(temp_chunk_file, "wb") as out:
                pickle.dump(current_chunk, out)
            print(f"Saved chunk: {temp_chunk_file} with {len(current_chunk)} entries")
            all_chunk_files.append(temp_chunk_file)
            current_chunk.clear()
            chunk_idx += 1

    print("Merging final chunks into single .pkl file...")
    final_list = []
    for chunk_file in all_chunk_files:
        with open(chunk_file, "rb") as cf:
            part = pickle.load(cf)
            final_list.extend(part)

    print(f"Final merged entry count: {len(final_list)}")
    with open(output_file, "wb") as fout:
        pickle.dump(final_list, fout)

    print(f"Saved final merged file to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g., BioLiP")
    parser.add_argument("--split", required=True, help="train / valid / test")
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    input_dir = f"./Dataset/{dataset}/intermediate_{split}"
    output_file = f"./Dataset/{dataset}/esm_{split}_{dataset}.pkl"
    temp_dir = f"./Dataset/{dataset}/temp_{split}_chunks"

    merge_in_segments(input_dir, output_file, temp_dir)
