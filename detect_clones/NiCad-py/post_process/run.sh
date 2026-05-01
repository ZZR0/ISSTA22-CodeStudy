: <<'COMMENT'
python 1_nicad_xml_to_jsonl.py \
    --xml ./input/azure-sdk-for-python_functions-clones-0.30-classes-withsource.xml \
    --out ./data/step1_nicad_azure_sim0.7.jsonl \
    --mode class \
    > ./data/step1_nicad_azure_sim0.7.log 2>&1

python 1_nicad_xml_to_jsonl.py \
    --xml ./input-bck/azure-sdk-for-python_functions-clones-0.30-classes-withsource.xml \
    --out ./data-bck/step1_nicad_azure_sim0.7.jsonl \
    --mode class \
    > ./data-bck/step1_nicad_azure_sim0.7.log 2>&1

python 2a_filter_out_test_fun.py \
    --input ./data/step1_nicad_azure_sim0.7.jsonl \
    --output ./data/step2a_nicad_azure_sim0.7.jsonl 

python 2b_filter_out_group.py \
    --input ./data/step2a_nicad_azure_sim0.7.jsonl \
    --output ./data/step2b_nicad_azure_sim0.7.jsonl \
    --min-size 1 --max-size 20 \
    > ./data/step2b_nicad_azure_sim0.7.log 2>&1

python 2c_remove_comment.py \
    --input ./data/step2b_nicad_azure_sim0.7.jsonl \
    --output ./data/step2c_nicad_azure_sim0.7.jsonl \
    > ./data/step2c_nicad_azure_sim0.7.log 2>&1

## python 2c_remove_comment.py \
##     --input ./data/small_input.jsonl \
##     --output ./data/small_output.jsonl \
##     > ./data/small.log 2>&1

python 2d_filter_out_init_fun.py \
    --input ./data/step2c_nicad_azure_sim0.7.jsonl \
    --output ./data/step2d_nicad_azure_sim0.7.jsonl \
    # > ./data/step2d_nicad_azure_sim0.7.log 2>&1
COMMENT

python display_clone_group_sizes.py \
    --input ./data/step2d_nicad_azure_sim0.7.jsonl 2>&1 \
    | tee ./data/display_clone_group_sizes1.log

python 3_gen_init_train_sample.py \
    --input ./data/step2d_nicad_azure_sim0.7.jsonl \
    --output ./data/step3_nicad_azure_sim0.7.jsonl \

python 4_gen_neg_clone_sample.py \
    --input ./data/step3_nicad_azure_sim0.7.jsonl \
    --out_txt ./data/step4_nicad_azure_sim0.7_neg_pairs.txt \
    --out_jsonl ./data/step4_nicad_azure_sim0.7_neg_pairs.jsonl \
    --out_html ./data/step4_nicad_azure_sim0.7_neg_pairs.html \
    --out_md ./data/step4_nicad_azure_sim0.7_neg_pairs.md \
    --seed 42 --verify --cleanup
echo '';echo '**************************************************************'
echo '**************************************************************';echo ''

python 5_gen_pos_clone_sample.py \
    --input ./data/step3_nicad_azure_sim0.7.jsonl \
    --out_txt ./data/step5_nicad_azure_sim0.7_pos_pairs.txt \
    --out_jsonl ./data/step5_nicad_azure_sim0.7_pos_pairs.jsonl \
    --out_html ./data/step5_display_nicad_azure_sim0.7_pos_pairs.html \
    --out_md ./data/step5_display_nicad_azure_sim0.7_pos_pairs.md \
    --seed 42 --verify --cleanup
echo '';echo '**************************************************************'
echo '**************************************************************';echo ''

python 6_split_combine_neg_pos_pairs.py \
    --neg ./data/step4_nicad_azure_sim0.7_neg_pairs.txt \
    --pos ./data/step5_nicad_azure_sim0.7_pos_pairs.txt \
    --out_dir ./data/azure \
    --seed 42 
echo '';echo '**************************************************************'
echo '**************************************************************';echo ''

python 7_gen_clone_bench.py \
  --input ./data/step3_nicad_azure_sim0.7.jsonl \
  --output ./data/azure/data.jsonl \
  --dedup
echo '';echo '**************************************************************'
echo '**************************************************************';echo ''
