mkdir -p ./saved_models/cs-java/cache_data
mkdir -p ./saved_models/cs-java/prediction
python run_gen.py    \
    --do_train \
    --do_eval \
    --do_eval_bleu \
    --do_test  \
    --save_last_checkpoints \
    --always_save_model   \
    --task translate \
    --sub_task cs-java \
    --model_type codet5 \
    --data_num -1    \
    --num_train_epochs 50 \
    --warmup_steps 1000 \
    --learning_rate 5e-5 \
    --patience 5   \
    --tokenizer_name=../../../CoTexT/cc/sentencepiece.model \
    --tokenizer_path=../../../CoTexT/cc/sentencepiece.model   \
    --model_name_or_path=../../../CoTexT/cc/ \
    --output_dir saved_models/cs-java/  \
    --summary_dir tensorboard   \
    --data_dir ../data/  \
    --cache_path saved_models/cs-java/cache_data \
    --res_dir saved_models/cs-java/prediction \
    --res_fn saved_models/cs-java/refine_cotext_base.txt   \
    --train_batch_size 12 \
    --eval_batch_size 4 \
    --max_source_length 320 \
    --max_target_length 256 \
    --gradient_accumulation_steps 2 \
    2>&1 | tee saved_models/cs-java/train.log
