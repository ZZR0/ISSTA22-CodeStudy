mkdir ./saved_models/
python run.py \
    --output_dir=./saved_models/ \
    --model_type=roberta \
    --model_name_or_path=../../../contracode/data/pretrain/ckpt_transformer_hybrid_pretrain_240k.pth \
    --tokenizer_name=../../../contracode/data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
    --pretrain_resume_encoder_name encoder_q \
    --do_test \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3 2>&1| tee ./saved_models/test.log
