import os
import re
import io
import sys
import json
import gzip
import copy
import tqdm
import random
import itertools
import multiprocessing
import argparse
from tree_sitter import Language, Parser
from io import StringIO
import tokenize
from parser import JAVA_AST as AST

random.seed(0)

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--max_holes', dest='max_holes', default=50, help='max number of holes to be inserted')
args = args_parser.parse_args()

def t_only_func_def(the_code, uid=1, all_sites=False):
    the_code = "public class Task {\n" + the_code + "\n}"
    changed = True
    the_ast = AST.build_ast(the_code)
    results = the_ast.get_func_def()
    func = ""
    for item in results:
        func += the_ast.source[item[0]:item[1]]
        func += " "
    site_map = {}
    func = func.strip()

    return changed, func, uid, site_map

def t_rename_func(the_code, uid=1, all_sites=False):
    """
    all_sites=True: a single, randomly selected, referenced field 
    (self.field in Python) has its name replaced by a hole
    all_sites=False: all possible fields are selected
    """
    the_code = "public class Task {\n" + the_code + "\n}"
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    func = the_ast.get_func_name()

    func_dict = AST.conv2dict(func)
    func_dict, code_list = AST.split_code(func_dict, the_code)
    
    site_map = {}

    for name in func_dict:
        for p in func_dict[name]:
            p, deep = p[0], p[1]

            site_map["function_{}".format(uid+count)] = (code_list[p], "transforms.RenameFields")
            code_list[p] = "function_{}".format(uid+count)
        count += 1
    
    the_code = "".join(code_list)
    the_code = the_code[20:-2]
    changed = count > 0
    return changed, the_code, uid+count-1, site_map


def t_rename_parameters(the_code, uid=1, all_sites=False):
    """
    Parameters get replaced by holes.
    """
    the_code = "public class Task {\n" + the_code + "\n}"
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    parameters = the_ast.get_parameters()

    parameters = [par.source for par in parameters]
    parameters = the_ast.get_selected_parameters(parameters)
    parameters_dict = AST.conv2dict(parameters)
    parameters_dict, code_list = AST.split_code(parameters_dict, the_code)
    
    site_map = {}

    for name in parameters_dict:
        for p in parameters_dict[name]:
            p, deep = p[0], p[1]

            site_map["arg_{}".format(uid+count)] = (code_list[p], "transforms.RenameParameters")
            code_list[p] = "arg_{}".format(uid+count)
        count += 1
    
    the_code = "".join(code_list)
    the_code = the_code[20:-2]

    changed = count > 0
    return changed, the_code, uid+count-1, site_map


def t_rename_local_variables(the_code, uid=1, all_sites=False):
    """
    Local variables get replaced by holes.
    """
    the_code = "public class Task {\n" + the_code + "\n}"
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    variables = the_ast.get_local_variables()

    variables = [par.source for par in variables]
    variables = the_ast.get_selected_variables(variables)
    variables_dict = AST.conv2dict(variables)
    variables_dict, code_list = AST.split_code(variables_dict, the_code)
    
    site_map = {}

    for name in variables_dict:
        for p in variables_dict[name]:
            p, deep = p[0], p[1]

            site_map["var_{}".format(uid+count)] = (code_list[p], "transforms.RenameLocalVariables")
            code_list[p] = "var_{}".format(uid+count)
        count += 1
    
    the_code = "".join(code_list)
    the_code = the_code[20:-2]
    
    changed = count > 0
    return changed, the_code, uid+count-1, site_map

def t_random_line_token(the_code, uid=1, all_sites=False):
    code = the_code.split('\n')
    random.shuffle(code)
    for idx in range(len(code)):
        line = code[idx]
        line = line.split(" ")
        random.shuffle(line)
        code[idx] = " ".join(line)

    return True, "\n".join(code), uid, {}

class t_seq(object):
    def __init__(self, transforms, all_sites):
        self.transforms = transforms
        self.all_sites = all_sites
    def __call__(self, the_ast, all_sites=False):
        did_change = False
        cur_ast = the_ast
        cur_idx = 0
        new_site_map = {}
        for t in self.transforms:
            changed, cur_ast, cur_idx, site_map = t(cur_ast, cur_idx+1, self.all_sites)
            if changed:
                did_change = True
                new_site_map.update(site_map)
        return did_change, cur_ast, cur_idx, new_site_map


def t_identity(the_code, all_sites=None):
    return True, the_code, 0, {}

def handle_replacement_tokens(line):
  new_line = line
  uniques = set()
  for match in re.compile('REPLACEME\d+').findall(line):
    uniques.add(match.strip())
  uniques = list(uniques)
  uniques.sort()
  uniques.reverse()
  for match in uniques:
    replaced = match.replace("REPLACEME", "_R") + '_'
    new_line = new_line.replace(match, replaced)
  return new_line


def process(item):
    (split, the_hash, og_code, as_json) = item
    transforms = []

    # transforms.append(('transforms.Combined', t_seq([t_rename_local_variables, t_rename_parameters, t_rename_fields, t_replace_true_false, t_insert_print_statements, t_add_dead_code], all_sites=True)))
    # transforms.append(('transforms.Insert', t_seq([t_insert_print_statements, t_add_dead_code], all_sites=True)))
    # transforms.append(('transforms.Replace', t_seq([t_rename_local_variables, t_rename_parameters, t_rename_fields, t_replace_true_false], all_sites=True)))
    transforms.append(('of', t_seq([t_only_func_def], all_sites=True)))
    transforms.append(('rf', t_seq([t_rename_func], all_sites=True)))
    transforms.append(('ra', t_seq([t_rename_parameters], all_sites=True)))
    transforms.append(('rv', t_seq([t_rename_local_variables], all_sites=True)))
    transforms.append(('rfra', t_seq([t_rename_func, t_rename_parameters], all_sites=True)))
    transforms.append(('rfrarv', t_seq([t_rename_func, t_rename_parameters, t_rename_local_variables], all_sites=True)))
    transforms.append(('rfraslst', t_seq([t_rename_func, t_rename_parameters, t_rename_local_variables, t_random_line_token], all_sites=True)))

    results = []
    for t_name, t_func in transforms:
        try:
            # print(t_func)
            changed, result, last_idx, site_map = t_func(
                og_code,
                all_sites=True
            )
            # result = handle_replacement_tokens(result)
            results.append((changed, split, t_name, the_hash, result, site_map, as_json)) 
        except Exception as ex:
            import traceback
            traceback.print_exc()
            results.append((False, split, t_name, the_hash, og_code, {}, as_json))
    return results


if __name__ == "__main__":
    print("Starting transform:")
    pool = multiprocessing.Pool(1)

    tasks = []

    print("  + Loading tasks...")
    splits = ['test.java-cs.txt.java']

    for split in splits:
        with open('./{}'.format(split)) as f:
            for idx, line in enumerate(f.readlines()):
                the_code = line.strip()
                the_code = AST.remove_comments_and_docstrings(the_code, "java")
                tasks.append((split, idx, the_code, {}))
    
    # tasks = tasks[:10]
    print("    + Loaded {} transform tasks".format(len(tasks)))
    results = pool.imap_unordered(process, tasks, 3000)

    print("  + Transforming in parallel...")
    names_covered = []
    all_sites = {}
    all_results = {}
    idx = 0
    for changed, split, t_name, the_hash, code, site_map, as_json in itertools.chain.from_iterable(tqdm.tqdm(results, desc="    + Progress", total=len(tasks))):
        idx += 1
        if not changed: continue

        as_json = copy.deepcopy(as_json)
        as_json["func"] = code
        if t_name not in all_sites:
            all_sites[t_name] = {split:{the_hash:site_map}}
            all_results[t_name] = {split:[as_json]}
        else:
            if split not in all_sites[t_name]:
                all_sites[t_name][split] = {the_hash:site_map}
                all_results[t_name][split] = [as_json]
            else:
                all_sites[t_name][split][the_hash] = site_map
                all_results[t_name][split] += [as_json]
    
    for t_name in all_results:
        for split in all_results[t_name]:
            if not os.path.exists('./'+t_name):
                os.makedirs('./'+t_name)
            with open('./{}/{}'.format(t_name, split), 'w') as f:
                for line in all_results[t_name][split]:
                    f.write(line["func"]+"\n")

    print("  + Transforms complete!")
