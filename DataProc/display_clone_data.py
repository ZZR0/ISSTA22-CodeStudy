import json
import os
import sys
import matplotlib.pyplot as plt
from collections import Counter

def plot_clone_distribution(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Cannot find file '{file_path}'.")
        return

    print(f"Reading data from: {file_path}")
    nclones_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Collect the "nclones" value from each JSON object (if present).
                if "nclones" in data:
                    nclones_list.append(data["nclones"])

        if not nclones_list:
            print("No 'nclones' information found in the data.")
            return

        counts = Counter(nclones_list)

        # Sort keys for x-axis and align counts for y-axis.
        sorted_x = sorted(counts.keys())
        sorted_y = [counts[x] for x in sorted_x]

        print("\n[Statistics]")
        print(f"{'nclones':<10} | {'Count':<10}")
        print("-" * 25)
        for x, y in zip(sorted_x, sorted_y):
            print(f"{x:<10} | {y:<10}")

        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_x, sorted_y, color='skyblue', edgecolor='black')

        plt.title('Distribution of Clone Group Sizes', fontsize=15)
        plt.xlabel('Number of Clones in Group (nclones)', fontsize=12)
        plt.ylabel('Number of Groups (Frequency)', fontsize=12)
        plt.xticks(sorted_x)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add labels above each bar showing the exact frequency.
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )

        output_file = 'clone_distribution_clone_func_small.png'
        plt.savefig(output_file)
        print(f"\nGraph saved to: {output_file}")
        print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Use the first CLI argument as input, otherwise fall back to the default path.
    # input_file = sys.argv[1] if len(sys.argv) > 1 else "data/nicad_camel_clone_func.jsonl"
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/nicad_camel_clone_func_small.jsonl"
    plot_clone_distribution(input_file)