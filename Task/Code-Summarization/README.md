# CodeXGLUE -- Code-To-Text

## Task Definition

The task is to generate natural language comments for a code, and evaluted by [smoothed bleu-4](https://www.aclweb.org/anthology/C04-1072.pdf) score.

## Dataset

The dataset we use comes from [CodeSearchNet](https://arxiv.org/pdf/1909.09436.pdf) and we filter the dataset as the following:

- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.

### Download data and preprocess

```shell
unzip dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

### Download data and preprocess in an online notebook(like Google Colab)

```shell
import os
!unzip dataset.zip
os.chdir("/content/dataset")
!wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
!wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
!wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
!wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
!wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
!wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

!unzip python.zip
!unzip java.zip
!unzip ruby.zip
!unzip javascript.zip
!unzip go.zip
!unzip php.zip
!rm *.zip
!rm *.pkl

!python preprocess.py
!rm -r */final
os.chdir("../")
```


### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

  - **repo:** the owner/repo

  - **path:** the full path to the original file

  - **func_name:** the function or method name

  - **original_string:** the raw string before tokenization or parsing

  - **language:** the programming language

  - **code/function:** the part of the `original_string` that is code

  - **code_tokens/function_tokens:** tokenized version of `code`

  - **docstring:** the top-level comment or docstring, if it exists in the original string

  - **docstring_tokens:** tokenized version of `docstring`

### Data Statistic

| Programming Language | Training |  Dev   |  Test  |
| :------------------- | :------: | :----: | :----: |
| Python               | 251,820  | 13,914 | 14,918 |
| PHP                  | 241,241  | 12,982 | 14,014 |
| Go                   | 167,288  | 7,325  | 8,122  |
| Java                 | 164,923  | 5,183  | 10,955 |
| JavaScript           |  58,025  | 3,885  | 3,291  |
| Ruby                 |  24,927  | 1,400  | 1,261  |

## Evaluator

We provide a script to evaluate predictions for this task, and report smoothed bleu-4 score.

### Example

```shell
python evaluator/evaluator.py evaluator/reference.txt < evaluator/predictions.txt
```

Total: 5
9.554726113590661

### Fine-tune and Evaluation
CodeBERT for example:
```shell
cd codebert
./run.sh
./eval.sh
```

The model files and result files are saved in `./codebert/saved_models` fold.