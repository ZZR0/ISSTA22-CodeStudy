# CodeXGLUE -- Clone Detection (POJ-104)


## Task Definition

Given a code and a collection of candidates as the input, the task is to return Top K codes with the same semantic. Models are evaluated by MAP@R score. MAP@R is defined as the mean of average precision scores, each of which is evaluated for retrieving R most similar samples given a query. For a code (query), R is the number of other codes in the same class, i.e. R=499 in this dataset.


## Dataset

We use [POJ-104](https://arxiv.org/pdf/1409.5718.pdf) dataset on this task.

### Download and Preprocess

1.Download dataset from [website](https://drive.google.com/file/d/0B2i-vWnOu7MxVlJwQXN6eVNONUU/view?usp=sharing) or run the following command:

```shell
cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU
tar -xvf programs.tar.gz
cd ..
```

2.Preprocess data

```shell
cd dataset
python preprocess.py
cd ..
```

### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

   - **code:** the source code
   - **label:** the number of problem that the source code solves
   - **index:** the index of example

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Problems | #Examples |
| ----- | --------- | :-------: |
| Train | 64        |  32,000   |
| Dev   | 16        |   8,000   |
| Test  | 24        |  12,000   |

## Evaluator

We provide a script to evaluate predictions for this task, and report MAP@R score.

### Example

Given a codes file evaluator/test.jsonl:

```bash
{"label": "65", "index": "0", "code": "function0"}
{"label": "65", "index": "1", "code": "function1"}
{"label": "65", "index": "2", "code": "function2"}
{"label": "66", "index": "3", "code": "function3"}
{"label": "66", "index": "4", "code": "function4"}
{"label": "66", "index": "5", "code": "function5"}
```

We first extract answers from codes file.

```she
python evaluator/extract_answers.py -c evaluator/test.jsonl -o evaluator/answers.jsonl 
```

The answers is:

```bash
cat evaluator/answers.jsonl 
{"index": "0", "answers": ["1", "2"]}
{"index": "1", "answers": ["0", "2"]}
{"index": "2", "answers": ["0", "1"]}
{"index": "4", "answers": ["3", "5"]}
{"index": "3", "answers": ["4", "5"]}
{"index": "5", "answers": ["4", "3"]}
```

Report MAP@R score

```shell
python evaluator/evaluator.py -a evaluator/answers.jsonl  -p evaluator/predictions.jsonl 
```

{'MAP@R': 0.6667}

### Input Predictions

For each index, return Top K (K=2 for this example, but K=499 in the task) codes. For example:

```shell
cat evaluator/predictions.jsonl 
{"index": "0", "answers": ["3", "2"]}
{"index": "1", "answers": ["0", "4"]}
{"index": "2", "answers": ["0", "1"]}
{"index": "4", "answers": ["1", "5"]}
{"index": "3", "answers": ["4", "2"]}
{"index": "5", "answers": ["4", "3"]}
```

### Fine-tune and Evaluation
CodeBERT for example:
```shell
cd codebert
./run.sh
./eval.sh
```

The model files and result files are saved in `./codebert/saved_models` fold.