mkdir ./saved_models/
python run.py \
    --output_dir=./saved_models/ \
    --model_type=t5 \
    --config_name=SEBIS/code_trans_t5_base_transfer_learning_pretrain  \
    --model_name_or_path=SEBIS/code_trans_t5_base_transfer_learning_pretrain  \
    --tokenizer_name=SEBIS/code_trans_t5_base_transfer_learning_pretrain  \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 2 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3 2>&1| tee ./saved_models/train.log
