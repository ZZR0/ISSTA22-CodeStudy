mkdir -p ./saved_models/
data_dir=../data/medium

python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --model_type t5 \
        --model_name_or_path SEBIS/code_trans_t5_base_transfer_learning_pretrain \
        --train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
        --dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
        --test_filename ../data/test.java-cs.txt.java,../data/test.java-cs.txt.cs \
        --output_dir ./saved_models/ \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 5 \
        --train_batch_size 8 \
        --eval_batch_size 16 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        2>&1 | tee ./saved_models/train.log