# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
from models import SearchModel
import logging
import argparse
import math
import numpy as np
import json
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
import time

from configs import add_args, set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_search_data, load_and_cache_clone_data
from models import get_model_size

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        code_ids = batch[0].to(args.device)
        nl_ids = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, code_vec, nl_vec = model(code_ids=code_ids, nl_ids=nl_ids)
            eval_loss += lm_loss.mean().item()
            code_vecs.append(code_vec.cpu().numpy())
            nl_vecs.append(nl_vec.cpu().numpy())
        nb_eval_steps += 1
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores=np.matmul(nl_vecs,code_vecs.T)
    ranks=[]
    for i in range(len(scores)):
        score=scores[i,i]
        rank=1
        for j in range(len(scores)):
            if i!=j and scores[i,j]>=score:
                rank+=1
        ranks.append(1/rank)    
    
    result = {
        "eval_loss": float(perplexity),
        "eval_mrr":float(np.mean(ranks))
    }

    if write_to_pred:
        sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
        indexs=[]
        urls=[]
        for example in eval_examples:
            indexs.append(example.idx)
            urls.append(example.url)
        with open(os.path.join(args.output_dir,"predictions.jsonl"),'w') as f:
            for index,url,sort_id in zip(indexs,urls,sort_ids):
                js={}
                js['url']=url
                js['answers']=[]
                for idx in sort_id[:100]:
                    js['answers'].append(indexs[int(idx)])
                f.write(json.dumps(js)+'\n')

    return result


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = SearchModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task, args=args)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_search_data(args, args.train_filename, pool, tokenizer, 'train',
                                                               is_sample=False)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader) // 5, 1)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_mrr = 0, 0
        not_mrr_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                code_ids, nl_ids = batch
                # pdb.set_trace()

                loss, _, _ = model(code_ids=code_ids, nl_ids=nl_ids)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += code_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0 and args.do_eval:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    eval_examples, eval_data = load_and_cache_search_data(args, args.dev_filename, pool, tokenizer,
                                                                         'valid', is_sample=True)

                    result = evaluate(args, model, eval_examples, eval_data)
                    eval_mrr = result['eval_mrr']

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_mrr', round(eval_mrr, 4), cur_epoch)

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    if True or args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_mrr > best_mrr:
                        not_mrr_inc_cnt = 0
                        logger.info("  Best mrr: %s", round(eval_mrr, 4))
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best mrr changed into %.4f\n" % (cur_epoch, round(eval_mrr, 4)))
                        best_mrr = eval_mrr
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-mrr')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ppl model into %s", output_model_file)
                    else:
                        not_mrr_inc_cnt += 1
                        logger.info("MRR does not increase for %d epochs", not_mrr_inc_cnt)
                        if not_mrr_inc_cnt > args.patience:
                            logger.info("Early stop as mrr do not increase for %d times", not_mrr_inc_cnt)
                            fa.write("[%d] Early stop as not_mrr_inc_cnt=%d\n" % (cur_epoch, not_mrr_inc_cnt))
                            is_early_stop = True
                            break

                model.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-mrr']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(file))
            else:
                model.load_state_dict(torch.load(file))

            if args.n_gpu > 1:
                # multi-gpu training
                model = torch.nn.DataParallel(model)

            eval_examples, eval_data = load_and_cache_search_data(args, args.test_filename, pool, tokenizer, 'test',
                                                                 False)

            result = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
            logger.info("  test_mrr=%.4f", result['eval_mrr'])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-mrr: %.4f\n" % (
                criteria, result['eval_mrr']))
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] mrr: %.4f\n\n" % (
                        criteria, result['eval_mrr']))
    fa.close()


if __name__ == "__main__":
    main()
