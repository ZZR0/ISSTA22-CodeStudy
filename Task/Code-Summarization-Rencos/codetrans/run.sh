lang=$1 #programming language
mkdir -p ./saved_models/$lang/
python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --model_type t5 \
        --model_name_or_path SEBIS/code_trans_t5_base_transfer_learning_pretrain \
        --train_filename ../dataset/$lang/train.jsonl \
        --dev_filename ../dataset/$lang/valid.jsonl \
        --test_filename ../dataset/$lang/test.jsonl \
        --output_dir ./saved_models/$lang \
        --max_source_length 256 \
        --max_target_length 128 \
        --beam_size 10 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 10 \
        2>&1 | tee ./saved_models/$lang/train.log