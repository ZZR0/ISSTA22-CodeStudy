# CodeXGLUE -- Text2Code Generation

Here are the dataset and pipeline for text-to-code generation task.

## Task Definition

Generate source code of class member functions in Java, given natural language description and class environment. Class environment is the programmatic context provided by the rest of the class, including other member variables and member functions in class. Models are evaluated by exact match and BLEU.

It's a challenging task because the desired code can vary greatly depending on the functionality the class provides. Models must (a) have a deep understanding of NL description and map the NL to environment variables, library API calls and user-defined methods in the class, and (b) decide on the structure of the resulting code.


## Dataset

### Concode dataset
We use concode dataset which is a widely used code generation dataset from Iyer's EMNLP 2018 paper [Mapping Language to Code in Programmatic Context](https://www.aclweb.org/anthology/D18-1192.pdf).

We have downloaded his published dataset and followed his preprocessed script. You can find the preprocessed data in `dataset/concode` directory.

Data statistics of concode dataset are shown in the below table:

|         |  #Examples  |
| ------- | :---------: |
|  Train  |   100,000   |
|   Dev   |    2,000    |
|  Test   |    2,000    |

### Data Format

Code corpus are saved in json lines format files. one line is a json object:
```
{
  "nl": "Increment this vector in this place. con_elem_sep double[] vecElement con_elem_sep double[] weights con_func_sep void add(double)",
  "code": "public void inc ( ) { this . add ( 1 ) ; }"
}
```

`nl` combines natural language description and class environment. Elements in class environment are seperated by special tokens like `con_elem_sep` and `con_func_sep`.

## Evaluator

We provide a script to evaluate predictions for this task, and report exact match and BLEU score. You can run the script like this:

```bash
python evaluator/evaluator.py -a=evaluator/answers.json -p=evaluator/predictions.txt
```

The outputs are:
```
BLEU: 20.21, EM: 17.0
```

The CodeBLEU score can be calculated by this [script](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU)

### Input Format

Answer file is in the same format of the dev set json lines file. A legal prediction file is expected to be a txt format file. It should have the same number of lines as answer file. Each line is the model prediction for the corresponding input in answer file. For example, one line in the answer file is:
```
{
  "nl": "Increment this vector in this place. con_elem_sep double[] vecElement con_elem_sep double[] weights con_func_sep void add(double)",
  "code": "public void inc ( ) { this . add ( 1 ) ; }"
}
```

And the corresponding line in your prediction file is:
```
public void inc ( ) { this . add ( 1 ) ; }
```


### Fine-tune and Evaluation
CodeBERT for example:
```shell
cd codebert
./run.sh
./eval.sh
```

The model files and result files are saved in `./codebert/saved_models` fold.