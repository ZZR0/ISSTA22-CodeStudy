# 1) Create a deterministic 10% subset of BigCloneBench (BCB)
#    - Uses a fixed random seed for reproducibility (seed=3)
#    - --mode balanced makes the sampled set label-balanced (0/1 = 50:50)
#    - Outputs (expected):
#        ../dataset/train_10percent.txt
#        ../dataset/valid_10percent.txt
python 1_sample_data.py --seed 3 --mode balanced

# 2) Build a mixed dataset: BCB(10%) + Other Domain (e.g., Camel)
#    - Merges clone-pair files (id1, id2, label) from:
#        * BCB (10% sampled):
#            - ../dataset/train_10percent.txt
#            - ../dataset/valid_10percent.txt
#        * Other domain (specified via --otherdomain_name, e.g., Camel):
#            - ../../../detect_clones/.../<otherdomain>/train.txt
#            - ../../../detect_clones/.../<otherdomain>/valid.txt
#
#    - Optionally prepares an OTHER-DOMAIN-ONLY test set for cross-domain evaluation:
#        * Input:
#            - ../../../detect_clones/.../<otherdomain>/test.txt
#        * Output (new file, no overwrite):
#            - ../dataset/test_<otherdomain>.txt
#
#    - Creates a combined code mapping (mix/data.jsonl) by concatenating:
#        * BCB mapping:
#            - ../dataset/data.jsonl
#        * Other-domain mapping:
#            - ../../../detect_clones/.../<otherdomain>/data.jsonl
#
#    - Prefixes all function IDs to avoid collisions and ensure resolvability:
#        * BCB IDs          -> bcb_<id>
#        * Other-domain IDs -> <otherdomain>_<id>
#
#    - Shuffles merged train / valid sets deterministically
#      using a fixed random seed (seed=3)
#
#    - Outputs (expected):
#        * ../dataset/train_mix.txt
#        * ../dataset/valid_mix.txt
#        * ../dataset/test_<otherdomain>.txt
#        * ../dataset/mix/data.jsonl
DATA_DIR="$HOME/nicad-clone-azure"

# Using 'python3' to be safe, and standardizing on spaces for args
python3 2_mix_data.py \
  --otherdomain_name azure \
  --train_data_file_bcb ../dataset/train_10percent.txt \
  --valid_data_file_bcb ../dataset/valid_10percent.txt \
  --train_data_file_more "${DATA_DIR}/train.txt" \
  --valid_data_file_more "${DATA_DIR}/valid.txt" \
  --test_data_file_otherdomain "${DATA_DIR}/test.txt" \
  --bcb_jsonl ../dataset/data.jsonl \
  --more_jsonl "${DATA_DIR}/data.jsonl" \
  --out_dir ../dataset/mix_azure \
  --out_train_mix ../dataset/train_mix_azure.txt \
  --out_valid_mix ../dataset/valid_mix_azure.txt \
  --seed 3

# Verify the azure mix output
python3 3_verify_data.py \
  --mapping_jsonl ../dataset/mix_azure/data.jsonl \
  --train_pairs ../dataset/train_mix_azure.txt \
  --valid_pairs ../dataset/valid_mix_azure.txt \
  --test_pairs ../dataset/test_azure.txt

# ################################################################3

# python 2_mix_data.py \
#   --otherdomain_name camel \
#   --train_data_file_bcb ../dataset/train_10percent.txt \
#   --valid_data_file_bcb ../dataset/valid_10percent.txt \
#   --train_data_file_more ../../../detect_clones/NiCad/post_process/data/java/camel/train.txt \
#   --valid_data_file_more ../../../detect_clones/NiCad/post_process/data/java/camel/valid.txt \
#   --test_data_file_otherdomain ../../../detect_clones/NiCad/post_process/data/java/camel/test.txt \
#   --bcb_jsonl ../dataset/data.jsonl \
#   --more_jsonl ../../../detect_clones/NiCad/post_process/data/java/camel/data.jsonl \
#   --seed 3

# python 3_verify_data.py --strict --otherdomain_name azure