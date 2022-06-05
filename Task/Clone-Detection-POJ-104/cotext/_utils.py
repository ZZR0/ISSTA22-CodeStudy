import json


def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str)
    source_ids = source_ids[:args.max_source_length-1] + [tokenizer.piece_to_id("</s>")]
    source_ids = source_ids + [tokenizer.piece_to_id("<pad>")] * (args.max_source_length-len(source_ids))
    
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str)
        target_ids = target_ids[:args.max_source_length-1] + [tokenizer.piece_to_id("</s>")]
        target_ids = target_ids + [tokenizer.piece_to_id("<pad>")] * (args.max_source_length-len(target_ids))
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    
    code1 = tokenizer.encode(source_str)
    code1 = code1[:args.max_source_length-1] + [tokenizer.piece_to_id("</s>")]
    code1 = code1 + [tokenizer.piece_to_id("<pad>")] * (args.max_source_length-len(code1))
    code2 = tokenizer.encode(target_str)
    code2 = code2[:args.max_source_length-1] + [tokenizer.piece_to_id("</s>")]
    code2 = code2 + [tokenizer.piece_to_id("<pad>")] * (args.max_source_length-len(code2))
    
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)

def convert_search_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        code_str = "{}: {}".format(args.task, example.code)
        nl_str = "{}: {}".format(args.task, example.nl)
    else:
        code_str = example.code
        nl_str = example.nl
    code_str = code_str.replace('</s>', '<unk>')
    nl_str = nl_str.replace('</s>', '<unk>')
    code_ids = tokenizer.encode(code_str)
    code_ids = code_ids[:args.max_source_length-1] + [tokenizer.piece_to_id("</s>")]
    code_ids = code_ids + [tokenizer.piece_to_id("<pad>")] * (args.max_source_length-len(code_ids))
    nl_ids = tokenizer.encode(nl_str)
    nl_ids = nl_ids[:args.max_source_length-1] + [tokenizer.piece_to_id("</s>")]
    nl_ids = nl_ids + [tokenizer.piece_to_id("<pad>")] * (args.max_source_length-len(nl_ids))
    
    assert code_ids.count(tokenizer.eos_token_id) == 1
    assert code_ids.count(tokenizer.eos_token_id) == 1

    return SearchInputFeatures(example_index, code_ids, nl_ids, example.url)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code_ids = tokenizer.encode(source_str)
    code_ids = code_ids[:args.max_source_length-1] + [tokenizer.piece_to_id("</s>")]
    code_ids = code_ids + [tokenizer.piece_to_id("<pad>")] * (args.max_source_length-len(code_ids))
    # code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code_ids, example.target)

class SearchInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 code_ids,
                 nl_ids,
                 url,

    ):
        self.example_id = example_id
        self.code_ids = code_ids
        self.nl_ids = nl_ids
        self.url=url

class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            line1 = replace_tokens(line1)
            line2 = replace_tokens(line2)
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            line1 = replace_tokens(line1)
            line2 = replace_tokens(line2)
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            code = x["code"].strip()
            code = replace_tokens(code)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=code
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens'])
            code = ' '.join(code.strip().split())
            code = replace_tokens(code)
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            code = replace_tokens(code)
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num, args):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/{}/data.jsonl'.format(args.test_type)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            code = replace_tokens(code)
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data

def replace_tokens(code):
    code = code.replace("\n", "NEW_LINE")
    code = code.replace("\t", "INDENT")
    code = code.replace("{", "OPEN_CURLY_TOKEN")
    code = code.replace("}", "CLOSE_CURLY_TOKEN")
    code = code.replace("<", "SMALLER_TOKEN")
    code = code.replace(">", "GREATER_TOKEN")
    code = code.replace("[", "OPEN_SQUARE_TOKEN")
    code = code.replace("]", "CLOSE_SQUARE_TOKEN")
    code = code.replace("$", "DOLLAR_TOKEN")
    return code

