mkdir -p ./saved_models/cache_data
python run_poj.py    \
    --do_eval \
    --do_eval_bleu \
    --do_test  \
    --save_last_checkpoints \
    --always_save_model   \
    --task poj \
    --sub_task none \
    --model_type codet5 \
    --data_num -1    \
    --num_train_epochs 2 \
    --warmup_steps 1000 \
    --learning_rate 2e-5 \
    --patience 2   \
    --tokenizer_name=../../../CoTexT/cc/sentencepiece.model \
    --tokenizer_path=../../../CoTexT/cc/sentencepiece.model   \
    --model_name_or_path=../../../CoTexT/cc/ \
    --output_dir saved_models/  \
    --summary_dir tensorboard   \
    --data_dir ../dataset/  \
    --cache_path saved_models/cache_data \
    --res_dir saved_models/prediction \
    --res_fn saved_models/search_codet5_base.txt   \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_source_length 400 \
    --max_target_length 400   \
    --gradient_accumulation_steps 1 \
    2>&1 | tee saved_models/test.log