import json
import os
import re
from io import StringIO
import tokenize
from nltk.tokenize import word_tokenize
from nltk.util import pr
from tqdm import tqdm

from tree_sitter import Language, Parser

#load parsers
lang = "java"
parsers={}        
LANGUAGE = Language('parser/my-languages.so', lang)
parser = Parser()
parser.set_language(LANGUAGE) 

def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def print_tree(tree):
    node_list = [tree]
    while node_list:
        node = node_list.pop()
        print(node.type, node.start_point, node.end_point)
        node_list.extend(node.children)

def clean_comment(comment):
    comment = comment.replace("/", "").replace("*", "")
    comment = comment.split(".")[0]
    comment = comment.split("?")[0]
    comment = comment.split("!")[0]

    tokens = word_tokenize(comment)
    return " ".join(tokens)

# {id, filepath, method_name, start_line, end_line, url}
def get_search_example(example):
    data_path = "/home/zzr/Neural-Code-Search-Evaluation-Dataset/projects/" 
    # print(example["id"])

    filepath = data_path+example["filepath"]
    # print(filepath)

    method_name = example["method_name"]
    start_line = example["start_line"]
    end_line = example["end_line"]
    comment = ""
    code = ""

    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                code_lines = f.readlines()
                code = ''.join(code_lines)
                tree = parser.parse(bytes(code,'utf8'))
                # print_tree(tree.root_node)
                query_str = """(
                                (comment)
                                (method_declaration
                                    name: ((identifier))
                                ) @func
                                )"""
                query = LANGUAGE.query(query_str)

                captures = query.captures(tree.root_node)
                for item in captures:
                    func = item[0]
                    if func.type != "method_declaration": continue
                    if func.start_point[0]+1 != start_line: continue
                    if func.end_point[0]+1 != end_line: continue
                    
                    comm = func.prev_named_sibling
                    if comm.type != "comment": continue
                    start_byte = comm.start_byte
                    end_byte = comm.end_byte

                    comment = code[start_byte:end_byte]
                    code = code_lines[start_line-1:end_line]
                    code = "".join(code)
                    # print(code)
                    break
        except UnicodeDecodeError:
            print("UnicodeDecodeError")
    
    code = remove_comments_and_docstrings(code, lang)
    original_comment = comment
    comment = clean_comment(comment)

    if comment and code:
        search_example = {
            "idx": example["id"],
            "url": example["url"],
            "language": "java",
            "function": code,
            "docstring_summary": comment,
            "docstring": original_comment,
            "identifier": method_name,
        }
        return search_example
    
    return None
# {url, sha, docstring_summary, language, parameters, argument_list,
# function_tokens, function, identifier, docstring, docstring_tokens,
# nwo, score, idx}

if __name__ == "__main__":

    with open("split.json", "r") as f:
        split = json.load(f)

    train_idx = list(map(int, split["train"]))[:1000]
    valid_idx = list(map(int, split["valid"]))[:1000]
    test_idx = list(map(int, split["test"]))[:1000]

    train_data, valid_data, test_data = [], [], []
    with open("search_corpus_1.jsonl", "r") as f:
        data = f.readlines()
        for idx in tqdm(train_idx):
            if idx <= len(data):
                line = data[idx-1]
                # print(idx)
                example = get_search_example(json.loads(line))
                if example is not None:
                    train_data.append(example)
        
        for idx in tqdm(valid_idx):
            if idx <= len(data):
                line = data[idx-1]
                example = get_search_example(json.loads(line))
                if example is not None:
                    valid_data.append(example)
        
        for idx in tqdm(test_idx):
            if idx <= len(data):
                line = data[idx-1]
                example = get_search_example(json.loads(line))
                if example is not None:
                    test_data.append(example)

    with open("search_corpus_2.jsonl", "r") as f:
        data = f.readlines()
        for idx in tqdm(train_idx):
            idx -= 2000000
            if idx > 0 and idx <= len(data):
                line = data[idx-1]
                # print(idx+2000000)
                example = get_search_example(json.loads(line))
                if example is not None:
                    train_data.append(example)
        
        for idx in tqdm(valid_idx):
            idx -= 2000000
            if idx > 0 and idx <= len(data):
                line = data[idx-1]
                example = get_search_example(json.loads(line))
                if example is not None:
                    valid_data.append(example)
        
        for idx in tqdm(test_idx):
            idx -= 2000000
            if idx > 0 and idx <= len(data):
                line = data[idx-1]
                example = get_search_example(json.loads(line))
                if example is not None:
                    test_data.append(example)

    with open("train.jsonl", "w") as f:
        for example in train_data:
            f.write(json.dumps(example)+'\n')
    
    with open("valid.jsonl", "w") as f:
        for example in valid_data:
            f.write(json.dumps(example)+'\n')

    with open("test.jsonl", "w") as f:
        for example in test_data:
            f.write(json.dumps(example)+'\n')

    
