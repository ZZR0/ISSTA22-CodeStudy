import os

# for item in ["codebert", "codegpt", "codetrans", "contracode", "graphcodebert"]:
#     cmd = """python calc_code_bleu.py \
#         --refs ../../{}/saved_models/test_-1.gold \
#         --hyp ../../{}/saved_models/test_-1.output \
#         --lang java"""
#     print(item)
#     os.system(cmd.format(item, item))


for item in ["codebert", "codegpt", "codetrans", "contracode", "graphcodebert"]:
    cmd = """python calc_code_bleu.py \
        --refs ../../{}/saved_models/small/test_-1.gold \
        --hyp ../../{}/saved_models/small/test_-1.output \
        --lang java"""
    print(item)
    os.system(cmd.format(item, item))