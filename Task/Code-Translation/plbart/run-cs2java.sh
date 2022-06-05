mkdir -p ./saved_models/cs2java/

python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --model_type bart \
        --model_name_or_path ../../../PLBART/pretrain/checkpoint_11_100000.pt \
        --tokenizer_name ../../../PLBART/sentencepiece/sentencepiece.bpe.model \
        --train_filename ../data/train.java-cs.txt.cs,../data/train.java-cs.txt.java \
        --dev_filename ../data/valid.java-cs.txt.cs,../data/valid.java-cs.txt.java \
        --test_filename ../data/test.java-cs.txt.cs,../data/test.java-cs.txt.java \
        --output_dir ./saved_models/cs2java/ \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 5 \
        --train_batch_size 8 \
        --eval_batch_size 16 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        2>&1 | tee ./saved_models/cs2java/train.log