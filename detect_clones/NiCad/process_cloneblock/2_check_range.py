import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract clone data with similarity < 80 and exclude the 'code' field.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output", required=True, help="Path to save the modified JSONL file")
    
    args = parser.parse_args()

    count = 0
    total_lines = 0
    try:
        with open(args.input, 'r', encoding='utf-8') as infile, \
             open(args.output, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                if not line.strip():
                    continue
                
                total_lines += 1
                data = json.loads(line)
                
                # Filter to only include lines with similarity below 80
                if data.get('similarity', 100.0) < 80.0:
                    
                    # Iterate through sources and remove the 'code' field safely
                    if 'sources' in data:
                        for source in data['sources']:
                            source.pop('code', None)
                            
                    # Write the modified dictionary back to the new file
                    outfile.write(json.dumps(data) + '\n')
                    count += 1

        print(f"Successfully processed '{args.input}'.")
        print(f"Total lines read: {total_lines}")
        print(f"Saved {count} stripped lines (similarity < 80) to '{args.output}'.")

    except FileNotFoundError:
        print(f"Error: The input file '{args.input}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON on line {total_lines + 1}. Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()