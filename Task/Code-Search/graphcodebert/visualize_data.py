import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

def draw_violinplot(heads, results, save_path='attn_rate.pdf'):
    # colors = ['C{}'.format(i) for i in range(len(projects))]
    # colors[-1] = 'black'
    # styles = ['' for _ in range(len(projects))]
    # styles[-1] = '-'
    # bottom, top = 0.5, 1

    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
    plt.violinplot(results, positions=[i for i in range(len(heads))], showmeans=True, showextrema=False)
    # plt.yticks(np.arange(bottom, top, step=0.1))
    plt.grid(axis="y", ls="--")

    results = np.transpose(np.array(results)).tolist()

    for data in zip(results):
        # data = [v * 100 for v in data]
        # ms = 6
        # if project == 'Mean':
        #     ms = 8
        plt.plot(heads, data[0], linestyle="", marker='.', c='black', ms=1)
        # plt.plot(labels, data, linestyle=s, marker=m)
    
    # plt.xlabel('Head')
    plt.ylabel('Attention Pay On Token (%)')
    # plt.legend(projects, loc=1)
    # plt.ylim(bottom, top)
    # plt.show()
    plt.savefig(save_path)

def plot_att_rate(token_att_list_all, token_list_all):
    heads = []
    code_attn_all_head = []
    dfg_attn_all_head = []
    for head in range(len(token_att_list_all[0])):
        heads.append("Head{}".format(head+1))
        code_attn, dfg_attn = [], []
        for attn_all, token_all in zip(token_att_list_all, token_list_all):
            attn = attn_all[head]
            token = token_all[head]
            for idx,word in enumerate(token):
                if word == "&ltunk&gt":
                    break
            if np.sum(attn) > 0:
                code_attn.append(100*np.sum(attn[:idx-1])/np.sum(attn))
                dfg_attn.append(100*np.sum(attn[idx-1:])/np.sum(attn))
        code_attn_all_head.append(code_attn)
        dfg_attn_all_head.append(dfg_attn)
    
    # code_attn_all_head = np.array(code_attn_all_head).T
    draw_violinplot(heads, code_attn_all_head)
    # print(code_attn_all_head[0])

def plot_att_avg_value(token_att_list_all, token_list_all):
    heads = []
    code_attn_all_head = []
    dfg_attn_all_head = []
    for head in range(len(token_att_list_all[0])):
        heads.append("Head{}".format(head+1))
        code_attn, dfg_attn = [], []
        for attn_all, token_all in zip(token_att_list_all, token_list_all):
            attn = attn_all[head]
            token = token_all[head]
            for idx,word in enumerate(token):
                if word == "&ltunk&gt":
                    break
            if np.sum(attn) > 0:
                code_attn.extend(attn[:idx-1])
                dfg_attn.extend(attn[idx-1:])
        token = round(np.mean(code_attn), 4)
        dfg = round(np.mean(dfg_attn), 4)
        code_attn_all_head.append(token)
        dfg_attn_all_head.append(dfg)
        print("Head{}   Token: {}   DFG: {}".format(head+1, token, dfg))
    
    # code_attn_all_head = np.array(code_attn_all_head).T
    # draw_violinplot(heads, code_attn_all_head)

def visualize(_id, masks, words, save_html=False, save_img=True):
    h5_string_list = list()
    h5_string_list.append('<div class="cam">')
    h5_string_list.append("<head><meta charset='utf-8'></head>")
    h5_string_list.append("Change Id: {}<br>".format(_id))

    save_path = './vis2/'
    for idx in range(len(masks)):
        line_mask = masks[idx]
        line_word = words[idx]
    #for line_mask, line_word, sent_attn in zip(masks, words, sent_attns):
        h5_string_list.append(
            '<font style="background: rgba(0, 0, 255, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>' % 1)
        for mask, word in zip(line_mask, line_word):
            h5_string_list.append('<font style="background: rgba(255, 0, 0, %f)">%s</font>' % (mask, word))
        h5_string_list.append('<br/>')
    h5_string_list.append('</div>')

    h5_string = ''.join(h5_string_list)

    h5_path = os.path.join(save_path, "{}.html".format(_id))
    with open(h5_path, "w") as h5_file:
        h5_file.write(h5_string)

with open('result_list.pkl', 'rb') as f:
    result_list = pickle.load(f)


token_list_all_head_all_result = []
att_list_all_head_all_result = []

for num_sample in range(len(result_list)):
    sample = result_list[num_sample]
    token_list_all = []
    token_att_list_all = []
    for num_head in range(len(sample['first'])):
        sample_code = sample['code'][1:-1]
        sample_str = sample['str'][3:-4]
        sample_w = sample['first'][num_head,1:-1]

        token_list = []
        for token in sample_code:
            token = token.replace(" ","")
            token = token.replace("Ġ" ," ")
            token_list.append(token.replace("<unk>" ,"&ltunk&gt"))

        sample_w = (sample_w - np.min(sample_w)) / (np.max(sample_w)-np.min(sample_w))
        token_list_all.append(token_list)
        token_att_list_all.append(sample_w)

    visualize(str(num_sample) + '_' + str(num_head), token_att_list_all, token_list_all)

    token_list_all_head_all_result.append(token_att_list_all)
    att_list_all_head_all_result.append(token_list_all)

plot_att_rate(token_list_all_head_all_result, att_list_all_head_all_result)
plot_att_avg_value(token_list_all_head_all_result, att_list_all_head_all_result)