# wandb offline
# python run.py train --run_name defect_detection --num_epochs 5 \
#                     --pretrain_resume_path ../../contracode/data/pretrain/ckpt_transformer_hybrid_pretrain_240k.pth \
#                     --pretrain_resume_encoder_name encoder_q
mkdir ./saved_models/
python run.py \
    --output_dir=./saved_models/ \
    --model_name_or_path=../../../contracode/data/pretrain/ckpt_transformer_hybrid_pretrain_240k.pth \
    --tokenizer_name=../../../contracode/data/codesearchnet_javascript/csnjs_8k_9995p_unigram_url.model \
    --pretrain_resume_encoder_name encoder_q \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 3  2>&1 | tee ./saved_models/train.log