mkdir ./saved_models/
python run.py \
    --output_dir=./saved_models \
    --model_type=bart \
    --model_name_or_path=../../../PLBART/pretrain/checkpoint_11_100000.pt \
    --tokenizer_name=../../../PLBART/sentencepiece/sentencepiece.bpe.model \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3 2>&1| tee ./saved_models/train.log
