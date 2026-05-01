mkdir ./saved_models_bcb/
python run.py \
    --output_dir=./saved_models_bcb/ \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../dataset/org/train.txt \
    --eval_data_file=../dataset/org/valid.txt \
    --test_data_file=../dataset/org/test.txt \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./saved_models_bcb/test_bcb.log