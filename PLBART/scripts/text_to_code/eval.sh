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

function generate () {

model=${SAVE_DIR}/checkpoint${1}.pt
FILE_PREF=${SAVE_DIR}/output_${1}
RESULT_FILE=${SAVE_DIR}/result.txt
GOUND_TRUTH_PATH=$PATH_2_DATA/valid.json

fairseq-generate $PATH_2_DATA/data-bin \
    --path $model \
    --task translation_from_pretrained_bart \
    --gen-subset valid \
    -t $TARGET -s $SOURCE \
    --scoring sacrebleu \
    --remove-bpe 'sentencepiece' \
    --batch-size 4 \
    --langs $langs \
    --beam 10 \
    --lenpen 1.0 > $FILE_PREF

cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp;

echo "CodeXGlue Evaluation Epoch $2" >> ${RESULT_FILE}
python evaluator.py \
    --expected $GOUND_TRUTH_PATH \
    --predicted ${FILE_PREF}.hyp \
    2>&1 | tee -a ${RESULT_FILE};
}

generate $2