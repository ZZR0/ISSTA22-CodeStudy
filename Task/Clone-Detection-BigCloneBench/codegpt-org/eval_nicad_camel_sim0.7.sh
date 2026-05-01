# Run test only on nicad_camel_sim0.7 test.txt. 
python run.py \
    --output_dir=./saved_models_bcb \
    --model_type=gpt2 \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --do_test \
    --train_data_file=../dataset/camel/train.txt \
    --eval_data_file=../dataset/camel/valid.txt \
    --test_data_file=../dataset/camel/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 \
    2>&1| tee ./saved_models_bcb/test_camel.log