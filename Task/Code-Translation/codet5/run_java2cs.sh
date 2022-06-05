mkdir -p ./saved_models/java-cs/cache_data
python run_gen.py    \
    --do_train \
    --do_eval \
    --do_eval_bleu \
    --do_test  \
    --save_last_checkpoints \
    --always_save_model   \
    --task translate \
    --sub_task java-cs \
    --model_type codet5 \
    --data_num -1    \
    --num_train_epochs 100 \
    --warmup_steps 1000 \
    --learning_rate 5e-5 \
    --patience 5   \
    --tokenizer_name=Salesforce/codet5-base \
    --tokenizer_path=../../../CodeT5/tokenizer/salesforce   \
    --model_name_or_path=Salesforce/codet5-base \
    --output_dir saved_models/java-cs/  \
    --summary_dir tensorboard   \
    --data_dir ../data/  \
    --cache_path saved_models/java-cs/cache_data \
    --res_dir saved_models/java-cs/prediction \
    --res_fn saved_models/java-cs/concode_codet5_base.txt   \
    --train_batch_size 12 \
    --eval_batch_size 4 \
    --max_source_length 320 \
    --max_target_length 256 \
    --gradient_accumulation_steps 2 \
    2>&1 | tee saved_models/java-cs/train.log
