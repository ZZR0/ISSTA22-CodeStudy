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
            echo "Syntax: bash run.sh GPU_ID SRC_LANG"
            echo
            echo "SRC_LANG  Language choices: [java, python, go, javascript, php, ruby]"
            echo
            exit;;
    esac
done

export CUDA_VISIBLE_DEVICES=$1

SOURCE=$2
TARGET=en_XX
PATH_2_DATA=${HOME_DIR}/data/codeXglue/code-to-text/${SOURCE}

echo "Source: $SOURCE Target: $TARGET"

SAVE_DIR=${CURRENT_DIR}/${SOURCE}_${TARGET}
mkdir -p ${SAVE_DIR}

if [[ "$SOURCE" =~ ^(ruby|javascript|go|php)$ ]]; then
    USER_DIR="--user-dir ${HOME_DIR}/source"
    TASK=translation_without_lang_token
else
    USER_DIR=""
    TASK=translation_from_pretrained_bart
fi

echo "USER_DIR: $USER_DIR"

function generate () {

model=${SAVE_DIR}/checkpoint_best.pt
FILE_PREF=${SAVE_DIR}/output
RESULT_FILE=${SAVE_DIR}/result.txt
GOUND_TRUTH_PATH=$PATH_2_DATA/test.jsonl

fairseq-generate $PATH_2_DATA/data-bin \
    --path $model \
    --task $TASK \
    --gen-subset test \
    -t $TARGET -s $SOURCE \
    --scoring sacrebleu \
    --remove-bpe 'sentencepiece' \
    --batch-size 4 \
    --langs $langs \
    --beam 10 > $FILE_PREF

cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[${TARGET}\]//g' > $FILE_PREF.hyp;
python evaluator.py $GOUND_TRUTH_PATH $FILE_PREF.hyp 2>&1 | tee ${RESULT_FILE};

}

generate
