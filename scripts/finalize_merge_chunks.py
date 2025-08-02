
import os
import pickle
import glob
import argparse

def finalize_merge_chunks(temp_dir, output_file):
    print(f"Finalizing merge from chunks in: {temp_dir}")
    chunk_files = sorted(glob.glob(os.path.join(temp_dir, "chunk_*.pkl")))

    total_entries = 0
    final_list = []

    for f in chunk_files:
        print(f"Loading chunk: {f}")
        with open(f, "rb") as pf:
            data = pickle.load(pf)
            final_list.extend(data)
            total_entries += len(data)

    with open(output_file, "wb") as fout:
        pickle.dump(final_list, fout)

    print(f"✅ Final merged file saved to: {output_file}")
    print(f"📦 Total entries: {total_entries}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g., BioLiP")
    parser.add_argument("--split", required=True, help="train / valid / test")
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    temp_dir = f"./Dataset/{dataset}/temp_{split}_chunks"
    output_file = f"./Dataset/{dataset}/esm_{split}_{dataset}.pkl"

    finalize_merge_chunks(temp_dir, output_file)
