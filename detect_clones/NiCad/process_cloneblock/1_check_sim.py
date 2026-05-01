import json
import argparse

def load_clone_classes(filepath):
    """Reads a JSONL file and extracts the classid, files, and their line ranges."""
    clone_classes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Map each file path to its specific line range
            file_ranges = {source['file']: source['range'] for source in data.get('sources', [])}
            
            clone_classes.append({
                'classid': data['classid'],
                'files': set(file_ranges.keys()),
                'file_ranges': file_ranges,
                'raw_data': data
            })
    return clone_classes

def main():
    parser = argparse.ArgumentParser(description="Find overlapping clone classes and extract their line ranges.")
    parser.add_argument("--func_file", required=True, help="Path to the filtered function-level JSONL file")
    parser.add_argument("--block_file", required=True, help="Path to the filtered block-level JSONL file")
    parser.add_argument("--output", required=False, help="Optional: Path to save the overlap results", default="overlap_results.jsonl")
    
    args = parser.parse_args()

    print(f"Loading function-level clones from: {args.func_file}")
    func_classes = load_clone_classes(args.func_file)
    
    print(f"Loading block-level clones from: {args.block_file}")
    block_classes = load_clone_classes(args.block_file)

    overlaps = []

    # Compare each function-level class against each block-level class
    for cf in func_classes:
        for cb in block_classes:
            # Check if all files in C_f exist in C_b
            if cf['files'].issubset(cb['files']) and len(cf['files']) > 0:
                
                # Extract the ranges for the shared files
                range_func_clone = {}
                range_block_clone = {}
                
                for file_path in cf['files']:
                    range_func_clone[file_path] = cf['file_ranges'][file_path]
                    range_block_clone[file_path] = cb['file_ranges'][file_path]

                overlaps.append({
                    "func_classid": cf['classid'],
                    "block_classid": cb['classid'],
                    "shared_files": list(cf['files']),
                    "range_func_clone": range_func_clone,
                    "range_block_clone": range_block_clone
                })

    print("-" * 40)
    print(f"Total function-level classes: {len(func_classes)}")
    print(f"Total block-level classes: {len(block_classes)}")
    print(f"Found {len(overlaps)} overlapping relationships.")
    print("-" * 40)

    # Save the results in JSONL format
    with open(args.output, 'w', encoding='utf-8') as out:
        for overlap in overlaps:
            out.write(json.dumps(overlap) + '\n')
            
    print(f"Detailed overlap mapping saved to: {args.output}")

if __name__ == "__main__":
    main()