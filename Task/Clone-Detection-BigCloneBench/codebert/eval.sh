
python run.py \
    --output_dir=./saved_models_bcb \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_test \
    --train_data_file=../dataset/org/train.txt \
    --eval_data_file=../dataset/org/valid.txt \
    --test_data_file=../dataset/org/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --predictions_file="predictions.txt" \
    --seed 123456 \
     2>&1| tee ./saved_models_bcb/test_bcb.log