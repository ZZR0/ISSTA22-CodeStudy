# TOKENIZER=microsoft/codebert-base   # solution default
TOKENIZER=roberta-base           # paper setting; it's the time for original paper reproduction.

mkdir -p ./saved_models/
python run.py \
  --output_dir=./saved_models/ \
  --model_type=roberta \
  --config_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --tokenizer_name=$TOKENIZER \
  --do_test \
  --train_data_file=../dataset/train_10percent.txt \
  --eval_data_file=../dataset/valid_10percent.txt \
  --test_data_file=../dataset/bcb/test.txt \
  --epoch 2 \
  --block_size 400 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 3 2>&1 | tee ./saved_models/test.log