mkdir -p ./saved_models/cache_data
mkdir -p ./saved_models/prediction

python run_gen.py    \
    --do_train \
    --do_eval \
    --do_eval_bleu \
    --do_test  \
    --save_last_checkpoints \
    --always_save_model   \
    --task summarize \
    --sub_task java \
    --model_type codet5 \
    --data_num -1    \
    --num_train_epochs 30 \
    --warmup_steps 1000 \
    --learning_rate 10e-5 \
    --patience 3   \
    --tokenizer_name=../../../CoTexT/cc/sentencepiece.model \
    --tokenizer_path=../../../CoTexT/cc/sentencepiece.model   \
    --model_name_or_path=../../../CoTexT/cc/ \
    --output_dir saved_models/  \
    --summary_dir tensorboard   \
    --data_dir ../dataset/  \
    --cache_path saved_models/cache_data \
    --res_dir saved_models/prediction \
    --res_fn saved_models/concode_codet5_base.txt   \
    --train_batch_size 12 \
    --eval_batch_size 3 \
    --max_source_length 256 \
    --max_target_length 128   \
    --gradient_accumulation_steps 2 \
    2>&1 | tee saved_models/train.log
