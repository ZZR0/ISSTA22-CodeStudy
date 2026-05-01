import argparse
import random
import os

def combine_and_shuffle(pos_path, neg_path, output_path):
    # Read files safely
    with open(pos_path, 'r', encoding='utf-8') as f:
        pos_lines = [line.strip() for line in f if line.strip()]
    with open(neg_path, 'r', encoding='utf-8') as f:
        neg_lines = [line.strip() for line in f if line.strip()]
    
    # Combine and shuffle
    combined = pos_lines + neg_lines
    random.shuffle(combined)
    
    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(f"{line}\n" for line in combined)
            
    return pos_lines, neg_lines

def main():
    parser = argparse.ArgumentParser(description="Combine and shuffle clone pair datasets.")
    parser.add_argument("--pos", required=True, help="Path to positive pairs file")
    parser.add_argument("--neg", required=True, help="Path to negative pairs file")
    parser.add_argument("--out", default="test.txt", help="Output file path (default: test.txt)")
    
    args = parser.parse_args()
    
    # Execute processing
    pos_lines, neg_lines = combine_and_shuffle(args.pos, args.neg, args.out)

    # Summary Output
    print("-" * 30)
    print("✅ Done")
    print(f"Neg: {len(neg_lines):>8}")
    print(f"Pos: {len(pos_lines):>8}")
    print(f"Total: {len(neg_lines) + len(pos_lines):>8}")
    print("-" * 30)
    print(f"Combined saved to: {args.out}")

if __name__ == "__main__":
    main()