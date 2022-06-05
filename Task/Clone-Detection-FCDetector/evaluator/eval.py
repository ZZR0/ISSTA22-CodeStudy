from sklearn.metrics import recall_score,precision_score,f1_score

def eval_file(pred_file, glod_file):
    preds, glods = [], []
    with open(pred_file, "r") as f:
        for line in f.readlines():
            line = [int(v) for v in line.strip().split()]
            preds.append(line)
    with open(glod_file, "r") as f:
        for line in f.readlines():
            line = [int(v) for v in line.strip().split()]
            glods.append(line)
    
    assert len(preds) == len(glods)
    for pred, glod in zip(preds, glods):
        assert pred[0] == glod[0]
        assert pred[1] == glod[1]

    y_true = [v[2] for v in glods]
    y_pred = [v[2] for v in preds]

    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))

    return f1, precision, recall

if __name__ == "__main__":
    pred_files = [
        "../codebert/saved_models/predictions.txt",
        "../codegpt/saved_models/predictions.txt",
        "../codet5/saved_models/predictions.txt",
        "../codetrans/saved_models/predictions.txt",
        "../contracode/saved_models/predictions.txt",
        "../cotext/saved_models/predictions.txt",
        "../graphcodebert/saved_models/predictions.txt",
        "../plbart/saved_models/predictions.txt",
        ]
    glod_file = "../dataset/test.txt"
    for pred_file in pred_files:
        print(pred_file)
        eval_file(pred_file, glod_file)
        print()

