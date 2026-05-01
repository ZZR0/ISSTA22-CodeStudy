from python_treesitter_parser import PythonTreeSitterParser
from util_ast_python import (
    iter_descendants, 
    method_name, 
    extract_rw_by_region,
    REG_PRE, REG_WITHIN, REG_POST
)

# 1. Define sample code with clear line numbers in mind
# Line 1:  class DataProcessor:
# Line 2:      factor: int = 2
# Line 3:  
# Line 4:      def calculate(self, items: list, offset: int):
# Line 5:          # PRE REGION
# Line 6:          total = 0
# Line 7:          results = []
# Line 8:  
# Line 9:          # WITHIN REGION (Clone Region: Lines 10-13)
# Line 10:         for x in items:
# Line 11:             val = x * self.factor + offset
# Line 12:             total += val
# Line 13:             results.append(val)
# Line 14: 
# Line 15:         # POST REGION
# Line 16:         final_score = total
# Line 17:         return results
source_code = """class DataProcessor:
    factor: int = 2

    def calculate(self, items: list, offset: int):
        # PRE REGION
        total = 0
        results = []

        # WITHIN REGION (Clone Region: Lines 10-13)
        for x in items:
            val = x * self.factor + offset
            total += val
            results.append(val)

        # POST REGION
        final_score = total
        return results
"""

def run_test():
    print("Initializing parser...")
    parser = PythonTreeSitterParser(source_code)
    
    # 2. Find the target method ('calculate')
    target_method = None
    for node in iter_descendants(parser.root):
        if node.type == "function_definition":
            if method_name(parser, node) == "calculate":
                target_method = node
                break
                
    if not target_method:
        print("Error: Could not find method 'calculate'")
        return

    # 3. Define the hypothetical "clone" region bounds
    clone_start_line = 10
    clone_end_line = 13
    
    print(f"\nAnalyzing data-flow for method: calculate()")
    print(f"Clone Region: Lines {clone_start_line} to {clone_end_line}")
    print("-" * 50)

    # 4. Extract Read/Write regions
    rw_regions = extract_rw_by_region(
        parser=parser,
        method_node=target_method,
        clone_start=clone_start_line,
        clone_end=clone_end_line,
        only_method_scope=True
    )

    # 5. Print the results nicely
    print("--- Scope Variables ---")
    print(f"Params: {rw_regions.params_in_method}")
    print(f"Locals: {rw_regions.locals_in_method}")
    print(f"Fields: {rw_regions.fields_in_class}")
    
    print("\n--- Reads (VR) ---")
    for region in [REG_PRE, REG_WITHIN, REG_POST]:
        print(f"  {region}: {sorted(list(rw_regions.vr[region]))}")
        
    print("\n--- Writes (VW) ---")
    for region in [REG_PRE, REG_WITHIN, REG_POST]:
        print(f"  {region}: {sorted(list(rw_regions.vw[region]))}")

if __name__ == "__main__":
    run_test()