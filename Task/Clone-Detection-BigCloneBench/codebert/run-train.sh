TOKENIZER=roberta-base           # paper setting; it's the time for original paper reproduction.

mkdir -p ./saved_models_combined_py/
python run.py \
  --output_dir=./saved_models_combined_py/ \
  --subsample_ratio 1.0 \
  --model_type=roberta \
  --config_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --tokenizer_name=$TOKENIZER \
  --do_train \
  --do_test \
  --train_data_file=../dataset/combined_py_plus_org10/train_mix.txt \
  --eval_data_file=../dataset/combined_py_plus_org10/valid_mix.txt \
  --test_data_file=../dataset/org/test.txt \
  --epoch 2 \
  --block_size 400 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 3 \
  2>&1 | tee ./saved_models_combined_py/train_combined_py_plus_org10_codebert.log


# do_train only, remove do_test. TOKENIZER=microsoft/codebert-base   # solution default
# TOKENIZER=roberta-base           # paper setting; it's the time for original paper reproduction.

# mkdir -p ./saved_models_bcb/
# python run.py \
#   --output_dir=./saved_models_bcb/ \
#   --model_type=roberta \
#   --config_name=microsoft/codebert-base \
#   --model_name_or_path=microsoft/codebert-base \
#   --tokenizer_name=$TOKENIZER \
#   --do_train \
#   --train_data_file=../dataset/org/train.txt \
#   --eval_data_file=../dataset/org/valid.txt \
#   --test_data_file=../dataset/org/test.txt \
#   --epoch 2 \
#   --block_size 400 \
#   --train_batch_size 16 \
#   --eval_batch_size 32 \
#   --learning_rate 5e-5 \
#   --max_grad_norm 1.0 \
#   --evaluate_during_training \
#   --seed 3 \
#   2>&1 | tee ./saved_models_bcb/train_bcb.log

# TOKENIZER=roberta-base           # paper setting; it's the time for original paper reproduction.

# mkdir -p ./saved_models_combined/
# python run.py \
#   --output_dir=./saved_models_combined/ \
#   --subsample_ratio 1.0 \
#   --model_type=roberta \
#   --config_name=microsoft/codebert-base \
#   --model_name_or_path=microsoft/codebert-base \
#   --tokenizer_name=$TOKENIZER \
#   --do_train \
#   --do_test \
#   --train_data_file=../dataset/combined_org10_plus_repos/train_mix.txt \
#   --eval_data_file=../dataset/combined_org10_plus_repos/valid_mix.txt \
#   --test_data_file=../dataset/org/test.txt \
#   --epoch 2 \
#   --block_size 400 \
#   --train_batch_size 16 \
#   --eval_batch_size 32 \
#   --learning_rate 5e-5 \
#   --max_grad_norm 1.0 \
#   --evaluate_during_training \
#   --seed 3 \
#   2>&1 | tee ./saved_models_combined/train_combined_org10_plus_repos_codebert.log