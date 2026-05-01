import json
import os
import sys

# Default file paths
# DEFAULT_INPUT = "data/nicad_camel_clone_func_small.jsonl"
# DEFAULT_OUTPUT = "data/nicad_camel_clone_data_small.jsonl"
DEFAULT_INPUT = "data/nicad_camel_clone_func.jsonl"
DEFAULT_OUTPUT = "data/nicad_camel_clone_data.jsonl"

def add_unique_ids(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"--- Processing: {input_file} ---")
    
    global_func_counter = 0
    total_groups = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    class_id = data.get("classid", "unknown") # Get Clone Group ID
                    sources = data.get("sources", [])
                    
                    # Iterate through each function source
                    for src in sources:
                        # Rule: {classid}_{global_uniq_id}
                        # Example: 1423_0, 1423_1, 99_2 ...
                        src["func_id"] = f"{class_id}_{global_func_counter}"
                        
                        global_func_counter += 1
                    
                    f_out.write(json.dumps(data) + "\n")
                    total_groups += 1
                    
                except json.JSONDecodeError:
                    print("Skipping invalid JSON line.")
                    continue
                    
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print(f"--- Completed ---")
    print(f"Processed {total_groups} clone groups.")
    print(f"Total functions processed: {global_func_counter}")
    print(f"Output saved to: {output_file}")

    print("\n[Verification Sample]")
    verify_output(output_file)

def verify_output(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                data = json.loads(first_line)
                if data.get('sources'):
                    first_src = data['sources'][0]
                    print(f"ClassID: {data['classid']}")
                    print(f"New ID Format: {first_src.get('func_id')}  <-- (Group_GlobalCount)")
    except Exception:
        pass

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    
    add_unique_ids(input_path, output_path)