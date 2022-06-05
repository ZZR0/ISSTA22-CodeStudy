from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu

gold_fn = "/data2/cg/CodeStudy/Task/Code-Refinement/codet5/saved_models/medium/prediction/test_best-bleu.gold"
output_fn = "/data2/cg/CodeStudy/Task/Code-Refinement/codet5/saved_models/medium/prediction/test_best-bleu.output"
lang = "java"
bleu = round(_bleu(gold_fn, output_fn), 2)
codebleu = round(calc_code_bleu.get_codebleu(gold_fn, output_fn, lang)*100, 2)

print(bleu, codebleu)