mkdir ./saved_models/
python run.py \
    --output_dir=./saved_models/ \
    --model_type=bart \
    --tokenizer_name=../../../PLBART/sentencepiece/sentencepiece.bpe.model \
    --model_name_or_path=../../../PLBART/pretrain/checkpoint_11_100000.pt \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ./saved_models/train.log
    # --no_cuda \
    # --fp16 \