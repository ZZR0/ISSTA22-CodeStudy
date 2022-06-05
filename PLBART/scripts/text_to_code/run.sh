#!/usr/bin/env bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;

PRETRAINED_MODEL_NAME=checkpoint_11_100000.pt
PRETRAIN=${HOME_DIR}/pretrain/${PRETRAINED_MODEL_NAME}
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX

while getopts ":h" option; do
    case $option in
        h) # display help
            echo
            echo "Syntax: bash run.sh GPU_ID"
            echo
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1

SOURCE=en_XX
TARGET=java
PATH_2_DATA=${HOME_DIR}/data/codeXglue/text-to-code/concode

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=${CURRENT_DIR}/concode
mkdir -p ${SAVE_DIR}


function fine_tune () {

OUTPUT_FILE=${SAVE_DIR}/finetune.log

fairseq-train $PATH_2_DATA/data-bin \
    --restore-file $PRETRAIN \
    --bpe 'sentencepiece' \
    --sentencepiece-model $SPM_MODEL \
    --langs $langs \
    --arch mbart_base \
    --layernorm-embedding \
    --task translation_from_pretrained_bart \
    --source-lang $SOURCE \
    --target-lang $TARGET \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.2 \
    --batch-size 8 \
    --update-freq 3 \
    --max-epoch 30 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay \
    --lr 5e-05 \
    --min-lr -1 \
    --warmup-updates 1000 \
    --max-update 200000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --seed 1234 \
    --log-format json \
    --log-interval 100 \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --reset-lr-scheduler \
    --eval-bleu \
    --eval-bleu-detok space \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-args '{"beam": 5}' \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --patience 10 \
    --ddp-backend no_c10d \
    --save-dir $SAVE_DIR \
    2>&1 | tee ${OUTPUT_FILE};

}


function generate () {

model=${SAVE_DIR}/checkpoint_best.pt
FILE_PREF=${SAVE_DIR}/output
RESULT_FILE=${SAVE_DIR}/result.txt
GOUND_TRUTH_PATH=$PATH_2_DATA/test.json

fairseq-generate $PATH_2_DATA/data-bin \
    --path $model \
    --task translation_from_pretrained_bart \
    --gen-subset test \
    -t $TARGET -s $SOURCE \
    --scoring sacrebleu \
    --remove-bpe 'sentencepiece' \
    --batch-size 4 \
    --langs $langs \
    --beam 10 \
    --lenpen 1.0 > $FILE_PREF

cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp;

echo "CodeXGlue Evaluation" >> ${RESULT_FILE}
python evaluator.py \
    --expected $GOUND_TRUTH_PATH \
    --predicted ${FILE_PREF}.hyp \
    2>&1 | tee -a ${RESULT_FILE};

echo "CodeBLEU Evaluation" >> ${RESULT_FILE}
cd ${HOME_DIR}/evaluation/CodeBLEU;
python calc_code_bleu.py \
    --refs ${SAVE_DIR}/grandtrue.txt \
    --hyp $FILE_PREF.hyp \
    --lang $TARGET \
    2>&1 | tee -a ${RESULT_FILE};
cd ${CURRENT_DIR};

}

fine_tune
# generate
