mkdir -p ./saved_models/cache_data
mkdir -p ./saved_models/prediction

python run_clone.py    \
    --do_train \
    --do_eval \
    --do_eval_bleu \
    --do_test  \
    --save_last_checkpoints \
    --always_save_model   \
    --task clone \
    --sub_task none \
    --model_type t5 \
    --data_num -1    \
    --num_train_epochs 2 \
    --warmup_steps 1000 \
    --learning_rate 5e-5 \
    --patience 2   \
    --tokenizer_name=SEBIS/code_trans_t5_base_transfer_learning_pretrain \
    --tokenizer_path=SEBIS/code_trans_t5_base_transfer_learning_pretrain   \
    --model_name_or_path=SEBIS/code_trans_t5_base_transfer_learning_pretrain \
    --output_dir saved_models/  \
    --summary_dir tensorboard   \
    --data_dir ../dataset/  \
    --cache_path saved_models/cache_data \
    --res_dir saved_models/prediction \
    --res_fn saved_models/clone_codet5_base.txt   \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --max_source_length 400 \
    --max_target_length 400   \
    --gradient_accumulation_steps 2 \
    --seed 2 \
    2>&1 | tee saved_models/train.log
