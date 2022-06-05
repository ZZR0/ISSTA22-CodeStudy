from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu

gold_fn = "/data2/cg/CodeStudy/Task/Code-Translation/codet5/saved_models/java-cs/prediction/test_best-bleu.gold"
output_fn = "/data2/cg/CodeStudy/Task/Code-Translation/codet5/saved_models/java-cs/prediction/test_best-bleu.output"
lang = "c_sharp"
bleu = round(_bleu(gold_fn, output_fn), 2)
codebleu = round(calc_code_bleu.get_codebleu(gold_fn, output_fn, lang)*100, 4)

print(bleu, codebleu)

gold_fn = "/data2/cg/CodeStudy/Task/Code-Translation/codet5/saved_models/cs-java/prediction/test_best-bleu.gold"
output_fn = "/data2/cg/CodeStudy/Task/Code-Translation/codet5/saved_models/cs-java/prediction/test_best-bleu.output"
lang = "java"
bleu = round(_bleu(gold_fn, output_fn), 2)
codebleu = round(calc_code_bleu.get_codebleu(gold_fn, output_fn, lang)*100, 4)

print(bleu, codebleu)