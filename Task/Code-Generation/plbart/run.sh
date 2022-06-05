mkdir -p ./saved_models/

python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --model_type bart \
        --model_name_or_path ../../../PLBART/pretrain/checkpoint_11_100000.pt \
        --tokenizer_name ../../../PLBART/sentencepiece/sentencepiece.bpe.model \
        --train_filename ../dataset/concode/train.json \
        --dev_filename ../dataset/concode/dev.json \
        --test_filename ../dataset/concode/test.json \
        --output_dir ./saved_models/ \
        --max_source_length 256 \
        --max_target_length 256 \
        --beam_size 10 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        --weight_decay=0.01 \
        2>&1 | tee ./saved_models/train.log