mkdir -p ./saved_models/$1
data_dir=../data/$1

python run.py \
        --do_test \
        --model_type roberta \
        --load_model_path ./saved_models/$1/checkpoint-best-bleu/pytorch_model.bin \
        --model_name_or_path microsoft/graphcodebert-base \
        --train_filename $data_dir/train.buggy-fixed.buggy,$data_dir/train.buggy-fixed.fixed \
        --dev_filename $data_dir/valid.buggy-fixed.buggy,$data_dir/valid.buggy-fixed.fixed \
        --test_filename $data_dir/test.buggy-fixed.buggy,$data_dir/test.buggy-fixed.fixed \
        --output_dir ./saved_models/$1/ \
        --max_source_length 256 \
        --max_target_length 256 \
        --beam_size 5 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        --lang java \
        2>&1 | tee ./saved_models/$1/test.log
