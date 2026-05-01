# python 1_check_sim.py --input data/camel_clone_block_level.jsonl --output data/filtered_camel_clone_block_level.jsonl
# python 1_check_sim.py --input data/camel_clone_func_level.jsonl --output data/filtered_camel_clone_func_level.jsonl

# python 1_check_sim.py \
#   --func_file data/filtered_camel_clone_func_level.jsonl \
#   --block_file data/filtered_camel_clone_block_level.jsonl \
#   --output data/overlap_results.json

# Strip code from the block-level file
python 2_check_range.py \
  --input data/camel_clone_block_level.jsonl \
  --output data/lightweight_camel_clone_block_level.jsonl

# Strip code from the func-level file
python 2_check_range.py \
  --input data/camel_clone_func_level.jsonl \
  --output data/lightweight_camel_clone_func_level.jsonl