mkdir ./saved_models/
python run_defect.py    \
    --do_train \
    --do_eval \
    --do_eval_bleu \
    --do_test  \
    --save_last_checkpoints \
    --always_save_model   \
    --task defect \
    --sub_task none \
    --model_type codet5 \
    --data_num 100    \
    --num_train_epochs 10 \
    --warmup_steps 1000 \
    --learning_rate 2e-5 \
    --patience 2   \
    --tokenizer_name=Salesforce/codet5-base \
    --tokenizer_path=../../../CodeT5/tokenizer/salesforce   \
    --model_name_or_path=Salesforce/codet5-base \
    --output_dir saved_models/  \
    --summary_dir tensorboard   \
    --data_dir ../dataset/  \
    --cache_path saved_models/cache_data \
    --res_dir saved_models/prediction \
    --res_fn saved_models/defect_codet5_base.txt   \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --max_source_length 512 \
    --max_target_length 3   \
    2>&1 | tee saved_models/train.log