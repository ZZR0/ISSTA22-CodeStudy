# CodeXGLUE -- Code2Code Translation

## Task Definition

Code translation aims to migrate legacy software from one programming language in a platform toanother.
In CodeXGLUE, given a piece of Java (C#) code, the task is to translate the code into C# (Java) version. 
Models are evaluated by BLEU scores, accuracy (exactly match), and [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/code-to-code-trans/CodeBLEU.MD) scores.

## Dataset

The dataset is collected from several public repos, including Lucene(http://lucene.apache.org/), POI(http://poi.apache.org/), JGit(https://github.com/eclipse/jgit/) and Antlr(https://github.com/antlr/).

We collect both the Java and C# versions of the codes and find the parallel functions. After removing duplicates and functions with the empty body, we split the whole dataset into training, validation and test sets.

### Data Format

The dataset is in the "data" folder. Each line of the files is a function, and the suffix of the file indicates the programming language.

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ------- | :-------: |
|  Train  |   10,300  |
|  Valid  |      500   |
|   Test  |    1,000  |

## Evaluator

We provide a script to evaluate predictions for this task, and report BLEU scores and accuracy (exactly math score).

### Example

```bash
python evaluator/evaluator.py -ref evaluator/references.txt -pre evaluator/predictions.txt
```

BLEU: 61.08, Acc: 50.0

### Fine-tune and Evaluation
CodeBERT for example:
```shell
cd codebert
./run.sh
./eval.sh
```

The model files and result files are saved in `./codebert/saved_models` fold.