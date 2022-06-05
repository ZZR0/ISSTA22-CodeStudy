import re
import pickle
import numpy as np
import os

# def indexall(str, key):
#     key_len = len(key)
#     indexs = []
#     for start in range(len(str)):
#         if start+key_len > len(str):break
#         if str[start:start+key_len] == key:
#             indexs.append(start)
    
#     return indexs

# # def get_spans_paire(l_idx, r_idx):
# #     count = 0
# #     for idx, l, r in enumerate(zip(l_idx, r_idx)):

# def get_spans_paires(line):
#     paires = []
#     l, r = [], []
#     paire = []
#     span = 0
#     for idx, w in enumerate(line):
#         if w == "<":
#             l.append(idx)
#         if w == ">":
#             r.append(idx)
        
#         if len(l) > 1:
#             l = [l[-1]]
#         if len(r) > 1:
#             r = [r[-1]]
        
#         if l and r and l[0]<r[0]:
#             paire.append([l[0], r[0]])
#             if line[l[0]+1] == "s":
#                 span += 1
#             if line[l[0]+1] == "/":
#                 span -= 1
#             l, r = [], []
            
        
#         if len(paire) > 1 and span==0:
#             paires.append(paire)
#             paire = []

#     return paires
# spans = []

# with open("3.span") as f:
#     spans = f.readlines()[1:-1]

# lines = [spans[i] for i in range(2,len(spans),4)]

# # print(lines)

# for line in lines[1:]:
#     line = re.sub("<td .* class=\"blob-code blob-code-inner js-file-line\">", "", line, count=0)
#     line = line[:-6]

#     paires = get_spans_paires(line)
#     last = 0
#     tokens = []
#     for pair in paires:
#         tokens.append(line[last:pair[0][0]])
#         if len(pair) == 4:
#             token = line[pair[1][1]+1:pair[-2][0]]
#         else:
#             token = line[pair[0][1]+1:pair[-1][0]]
#         tokens.append(token)
#         last = pair[-1][1]+1
#     tokens.append(line[last:])

#     print("".join(tokens))

def visualize(_id, masks, words, keys, save_html=False, save_img=True):
    h5_string_list = list()
    h5_string_list.append('<tbody>\n')

    save_path = './'

    # h5_string_list.append("""<tr>
    #     <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
    #     <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class="pl-s">\"""Fetch insitiute and case objects.\"""</span></td>
    #   </tr>""")
    for idx in range(len(masks)):
        line_mask = masks[idx]
        line_word = words[idx]
        line_key = keys[idx]
        h5_string_list.append('<tr>\n')

        h5_string_list.append("""<td id="L{}" class="blob-num js-line-number" data-line-number="{}"></td>\n""".format(idx+1, idx+1))
        h5_string_list.append("""<td id="LC2" class="blob-code blob-code-inner js-file-line">""")
        for mask, word, key in zip(line_mask, line_word, line_key):
            if key.replace(" ","") == "":
                h5_string_list.append('<span style="background: rgba(255, 0, 0, %f)">%s</span>' % (mask, word))
            else:
                h5_string_list.append('<span style="background: rgba(255, 0, 0, %f)">%s</span>' % (mask, word))
                # h5_string_list.append('<span style="background: rgba(255, 0, 0, %f)" class="%s">%s</span>' % (mask, key, word))

        h5_string_list.append("</td>\n")
        h5_string_list.append('</tr>\n')
    h5_string_list.append('</tbody>')

    h5_string = ''.join(h5_string_list)

    h5_path = os.path.join(save_path, "{}.html".format(_id))
    with open(h5_path, "w") as h5_file:
        h5_file.write(h5_string)

with open('result_list.pkl', 'rb') as f:
    result_list = pickle.load(f)

sample = result_list[3]
num_head = 8
sample_code = sample['code'][1:-1]
sample_str = sample['str'][3:-4]
sample_w = sample['first'][num_head,1:-1]

token_list = []
for token in sample_code:
    token = token.replace(" ","")
    token = token.replace("Ġ" ," ")
    token = token.replace("<unk>" ," ")
    token_list.append(token)

sample_w = (sample_w - np.min(sample_w)) / (np.max(sample_w)-np.min(sample_w))

assert len(token_list) == len(sample_w)
print(token_list)
# print(sample_w)

token_list=[
    ['def', " ", 'Fun','c', '(', 'arg', '_', '0', " ", ',', 'arg', '_', '1', " ", ',', 'arg', '_', '2', '=', 'None', ')', ':'],
    ["  ", 'arg', '_', '3', " ", '=', " ", 'arg', '_', '0', '.', 'institute', '(', 'arg', '_', '1', ')'],
    ["  ", 'if', " ", 'arg', '_', '3', " ", 'is', " ", 'None', " ", 'and', " ", 'arg', '_', '1', " ", '!=', " ", "'", 'f', 'av', 'icon', '.', 'ico', "'", ':'],
    ["    ", 'flash', '(', '"', 'Can', "'t", " ", 'find', " ", 'institute', ':', " ", '{}', '"', '.', 'format', '(', 'arg', '_', '1', ')', ',', " ", "'", 'warning', "'", ')'],
    ["    ", 'return', " ", 'abort', '(', '404', ')'],
    ["  ", 'if', " ", 'arg', '_', '2', ':'],
    ["    ", 'if', " ", 'arg', '_', '2', ':'],
    ["      ", 'arg', '_', '4', " ", '=', " ", 'arg', '_', '0', '.', 'case', '(', 'arg', '_', '1', '=', 'arg', '_', '1', ',', " ", 'display', '_', 'name', '=', 'arg', '_', '2', ')'],
    ["      ", 'if', " ", 'arg', '_', '4', " ", 'is', " ", 'None', ':'],
    ["        ", 'return', " ", 'abort', '(', '404', ')'],
    ["  ", 'if', " ", 'not', " ", 'current', '_', 'user', '.', 'is', '_', 'admin', ':'],
    ["    ", 'if', " ", 'arg', '_', '1', " ", 'not', " ", 'in', " ", 'current', '_', 'user', '.', 'instit', 'utes', ':'],
    ["      ", 'if', " ", 'not', " ", 'arg', '_', '2', " ", 'or', " ", 'not', " ", 'any', '(', 'arg', '_', '5', " ", 'in', " ", 'arg', '_', '4', '[', "'", 'coll', 'abor', 'ators', "'", ']'], 
    ["              ", 'for', " ", 'arg', '_', '5', " ", 'in', " ", 'current', '_', 'user', '.', 'instit', 'utes', ')', ':'],
    ["        ", 'flash', '(', '"', 'You', " ", 'don', "'t", " ", 'have', " ", 'acc', 'cess', " ", 'to', ':', " ", '{}', '"'], 
    ["              ", '.', 'format', '(', 'arg', '_', '1', ')', ',', "'", 'danger', "'", ')'],
    ["        ", 'return', " ", 'abort', '(', '403', ')'],
    ["  ", 'if', " ", 'arg', '_', '2', ':'," ", 'return', " ", 'arg', '_', '3', ',', " ", 'arg', '_', '4'],
    ["  ", 'else', ':'," ", 'return', " ", 'arg', '_', '3'],
    ['&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt'], 
    ['&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt'], 
    ['&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt', '&ltunk&gt']
]

key_list=[
    ['pl-k', " ", 'pl-v','pl-v', ' ', 'pl-s1', 'pl-s1', 'pl-s1', " ", '', 'pl-s1', 'pl-s1', 'pl-s1', " ", '', 'pl-s1', 'pl-s1', 'pl-s1', 'pl-c1', 'pl-c1', '', ''],
    ["    ", 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', " ", 'pl-s1', 'pl-s1', 'pl-s1', '', 'pl-en', '', 'pl-s1', 'pl-s1', 'pl-s1', ''],
    ["    ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', " ", 'pl-c1', " ", 'pl-c1', " ", 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', " ", "pl-s", 'pl-s', 'pl-s', 'pl-s', 'pl-s', 'pl-s', "pl-s", ''],
    ["    ", 'pl-en', '', 'pl-s', 'pl-s', "pl-s", "pl-s", 'pl-s', "pl-s", 'pl-s', 'pl-s', "pl-s", 'pl-s', 'pl-s', '', 'pl-en', '(', 'pl-s1', 'pl-s1', 'pl-s1', '', '', " ", "pl-s", 'pl-s', "pl-s", ''],
    ["        ", 'pl-k', " ", 'pl-en', '', 'pl-c1', ''],
    ["    ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1', ''],
    ["        ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1', ''],
    ["            ", 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', " ", 'pl-s1', 'pl-s1', 'pl-s1', '', 'pl-en', '', 'pl-s1', 'pl-s1', 'pl-s1', 'pl-c1', 'pl-s1', 'pl-s1', 'pl-s1', '', " ", 'pl-s1', 'pl-s1', 'pl-s1', 'pl-c1', 'pl-s1', 'pl-s1', 'pl-s1', ''],
    ["            ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', " ", 'pl-c1', ''],
    ["                ", 'pl-k', " ", 'pl-en', '', 'pl-c1', ''],
    ["    ", 'pl-k', " ", 'pl-c1', " ", 'pl-s1', 'pl-s1', 'pl-s1', '', 'pl-s1', 'pl-s1', 'pl-s1', ':'],
    ["        ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', " ", 'pl-c1', " ", 'pl-s1', 'pl-s1', 'pl-s1', '', 'pl-s1', 'pl-s1', ''],
    ["            ", 'pl-k', " ", 'pl-c1', " ", 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', " ", 'pl-c1', " ", 'pl-en', '', 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', " ", 'pl-s1', 'pl-s1', 'pl-s1', '', "pl-s", 'pl-s', 'pl-s', 'pl-s', "pl-s", ''],
    ["                                    ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1', " ", 'pl-c1', "" 'pl-s1', 'pl-s1', 'pl-s1', '', 'pl-s1', 'pl-s1', '', ''],
    ["                ", 'pl-en', '', 'pl-s', 'pl-s', "pl-s", 'pl-s', "pl-s", "pl-s", 'pl-s', "pl-s", 'pl-s', 'pl-s', "pl-s", 'pl-s', 'pl-s', "pl-s", 'pl-s', 'pl-s'], 
    [" ", '', 'pl-en', '', 'pl-s1', 'pl-s1', 'pl-s1', ')', ',', "pl-s", 'pl-s', "pl-s", ''],
    ["                ", 'pl-k', " ", 'pl-en', '', 'pl-c1', ''],
    ["    ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1', ''," ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1', '', " ", 'pl-s1', 'pl-s1', 'pl-s1'],
    ["    ", 'pl-k', ''," ", 'pl-k', " ", 'pl-s1', 'pl-s1', 'pl-s1'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], 
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
]

wight_list = []
idx = 0
for line in token_list:
    # line = len(line)
    wight_line = []
    for token in line:
        if token.replace(" ","") == "":
            wight_line.append(0)
        else:
            wight_line.append(sample_w[idx])
            idx += 1
    wight_list.append(wight_line)
    assert len(wight_list[-1]) == len(line)

visualize(3, wight_list, token_list, key_list)

# print(token_list)
# print(sample_w)

