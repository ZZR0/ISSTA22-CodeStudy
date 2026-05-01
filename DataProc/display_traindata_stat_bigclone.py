import os
import sys

# Default input file path
DEFAULT_INPUT = "./data/train.txt"

def analyze_train_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"--- Analyzing Training Data: {file_path} ---")

    # Sets to store unique function IDs
    funcs_label_0 = set() # Functions appearing in Label 0 (Non-clone)
    funcs_label_1 = set() # Functions appearing in Label 1 (Clone)
    
    pairs_label_0 = 0
    pairs_label_1 = 0
    total_lines = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Split by whitespace or tab
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                # Format: ID1  ID2  Label
                id1, id2, label = parts[0], parts[1], parts[2]
                
                total_lines += 1
                
                if label == '0':
                    pairs_label_0 += 1
                    funcs_label_0.add(id1)
                    funcs_label_0.add(id2)
                elif label == '1':
                    pairs_label_1 += 1
                    funcs_label_1.add(id1)
                    funcs_label_1.add(id2)

    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Check intersection (functions appearing in both Label 0 and 1)
    overlap_funcs = funcs_label_0.intersection(funcs_label_1)
    
    # Total unique functions (Union)
    all_unique_funcs = funcs_label_0.union(funcs_label_1)

    # --- Print Results ---
    print(f"\n=== Statistics Report ===")
    print(f"Total Pairs (Lines): {total_lines}")
    print(f"Total Unique Functions: {len(all_unique_funcs)}")

    print(f"\n[Label 0 (Non-Clone)]")
    print(f"  - Pairs Count: {pairs_label_0}")
    print(f"  - Unique Functions Involved: {len(funcs_label_0)}")

    print(f"\n[Label 1 (Clone)]")
    print(f"  - Pairs Count: {pairs_label_1}")
    print(f"  - Unique Functions Involved: {len(funcs_label_1)}")

    print(f"\n[Overlap]")
    print(f"  - Functions appearing in BOTH labels: {len(overlap_funcs)}")
    if len(overlap_funcs) > 0:
        print(f"    (Note: {len(overlap_funcs)} functions are used as both positive and negative samples.)")

if __name__ == "__main__":
    # Use the file from command line args if present, otherwise use default
    target_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    analyze_train_data(target_file)