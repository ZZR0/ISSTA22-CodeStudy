# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import random
from tqdm import tqdm
import re
from io import StringIO
import  tokenize

random.seed(0)

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

def files(path):
    g = os.walk(path) 
    file=[]
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            file.append(os.path.join(path, file_name))
    return file

data = {}

cont=0
with open("data.jsonl",'w') as f:
    for i in tqdm(range(0,65),total=65):
        items=files("ProgramData/{}".format(i))
        for item in items:
            js={}
            js['label']=item.split('/')[1]
            js['index']=str(cont)
            js['idx']=str(cont)

            data[cont] = js['label']
            code = open(item,encoding='latin-1').read()
            code = remove_comments_and_docstrings(code, "c")
            code = code.replace("\n", " ").replace("\t", " ")
            code = " ".join(code.split())
            js['code']=code
            js['func']=code
            f.write(json.dumps(js)+'\n')
            cont+=1

data0 = []
data1 = []

for i in data.keys():
    for j in data.keys():
        if data[i] != data[j]:
            data0.append("{}\t{}\t0\n".format(i,j))
        else:
            data1.append("{}\t{}\t1\n".format(i,j))

random.shuffle(data0)
random.shuffle(data1)

data = data1 + data0[:len(data1)]

random.shuffle(data)

with open("train.txt", "w") as f:
    f.writelines(data[:int(0.8*len(data))])

with open("test.txt", "w") as f:
    f.writelines(data[int(0.8*len(data)):])

with open("valid.txt", "w") as f:
    f.writelines(data[int(0.9*len(data)):])