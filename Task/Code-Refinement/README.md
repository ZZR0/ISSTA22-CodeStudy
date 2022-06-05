# CodeXGLUE -- Code Refinement

## Task Definition

Code refinement aims to automatically fix bugs in the code, which can contribute to reducing the cost of bug-fixes for developers.
In CodeXGLUE, given a piece of Java code with bugs, the task is to remove the bugs to output the refined code. 
Models are evaluated by BLEU scores, accuracy (exactly match) and [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/code-to-code-trans/CodeBLEU.MD).

## Dataset

We use the dataset released by this paper(https://arxiv.org/pdf/1812.08693.pdf). The source side is a Java function with bugs and the target side is the refined one. 
All the function and variable names are normalized. Their dataset contains two subsets ( i.e.small and medium) based on the function length.

### Data Format

The dataset is in the "data" folder. Each line of the files is a function.

### Data Statistics

Data statistics of this dataset are shown in the below table:

|         | #Examples | #Examples |
| ------- | :-------: | :-------: |
|         |   Small   |   Medium  |
|  Train  |   46,680  |   52,364  |
|  Valid  |    5,835  |    6,545  |
|   Test  |    5,835  |    6,545  |

## Evaluator

We provide a script to evaluate predictions for this task, and report BLEU scores and accuracy (exactly math score).

### Example

```bash
python evaluator/evaluator.py -ref evaluator/references.txt -pre evaluator/predictions.txt
```

BLEU: 79.03, Acc: 40.0

### Fine-tune and Evaluation
CodeBERT for example:
```shell
cd codebert
./run.sh
./eval.sh
```

The model files and result files are saved in `./codebert/saved_models` fold.