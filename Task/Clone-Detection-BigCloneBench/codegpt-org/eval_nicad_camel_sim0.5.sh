# Run test only on nicad_camel_sim0.5 test.txt. Don't forget to change the "predictions_camel_sim0.5_test.txt" in run.py
python run.py \
    --output_dir=./saved_models_bcb \
    --model_type=gpt2 \
    --config_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --do_test \
    --train_data_file=../dataset/camel-sim0.5/train.txt \
    --eval_data_file=../dataset/camel-sim0.5/valid.txt \
    --test_data_file=../dataset/camel-sim0.5/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --predictions_file=predictions_camel_sim0.5_test.txt \
    --seed 123456 \
    2>&1| tee ./saved_models_bcb/test_camel_0.5.log