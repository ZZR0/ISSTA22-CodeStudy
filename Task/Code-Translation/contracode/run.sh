mkdir -p ./saved_models/java2cs/

python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --model_name_or_path ../../../contracode/data/pretrain/ckpt_transformer_hybrid_pretrain_240k.pth \
        --tokenizer_name ../../../contracode/data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
        --pretrain_resume_encoder_name encoder_q \
        --train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
        --dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
        --test_filename ../data/test.java-cs.txt.java,../data/test.java-cs.txt.cs \
        --output_dir ./saved_models/java2cs/ \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 5 \
        --train_batch_size 8 \
        --eval_batch_size 16 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        2>&1 | tee ./saved_models/java2cs/train.log