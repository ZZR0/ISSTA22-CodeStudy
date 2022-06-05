# Clone Detection (FCDetector)

## Task Definition

Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.

## Dataset

The dataset we use is [FCDetector](https://github.com/shiyy123/FCDetector) and filtered following the paper [Functional code clone detection with syntax and semantics fusion learning.](https://qingkaishi.github.io/public_pdfs/ISSTA20-CCD.pdf).

### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the function

   - **idx:** index of the example

2. train.txt/valid.txt/test.txt provide examples, stored in the following format:    idx1	idx2	label

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  240,160  |
| Dev   |  30,020  |
| Test  |  60,040  |

## Evaluator

We provide a script to evaluate predictions for this task, and report F1 score

### Example

```bash
python evaluator/evaluator.py -a evaluator/answers.txt -p evaluator/predictions.txt
```

{'Recall': 0.25, 'Prediction': 0.5, 'F1': 0.3333333333333333}

### Input predictions

A predications file that has predictions in TXT format, such as evaluator/predictions.txt. For example:

```b
13653451	21955002	0
1188160	8831513	1
1141235	14322332	0
16765164	17526811	1
```

### Fine-tune and Evaluation
CodeBERT for example:
```shell
cd codebert
./run.sh
./eval.sh
```

The model files and result files are saved in `./codebert/saved_models` fold.