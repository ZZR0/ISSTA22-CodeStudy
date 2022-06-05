mkdir -p ./saved_models/
data_dir=../data/medium

python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --model_name_or_path ../../../contracode/data/pretrain/ckpt_transformer_hybrid_pretrain_240k.pth \
        --tokenizer_name ../../../contracode/data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
        --pretrain_resume_encoder_name encoder_q \
        --train_filename $data_dir/train.buggy-fixed.buggy,$data_dir/train.buggy-fixed.fixed \
        --dev_filename $data_dir/valid.buggy-fixed.buggy,$data_dir/valid.buggy-fixed.fixed \
        --test_filename $data_dir/test.buggy-fixed.buggy,$data_dir/test.buggy-fixed.fixed \
        --output_dir ./saved_models/ \
        --max_source_length 256 \
        --max_target_length 256 \
        --beam_size 5 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        2>&1 | tee ./saved_models/train.log