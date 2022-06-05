#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/12/22 19:50
# @Author   : Xiang Ling
# @File     : config.py
# @Lab      : nesa.zju.edu.cn
# ************************************
import argparse

parser = argparse.ArgumentParser(description="DGMS")

parser.add_argument('--data_dir', type=str, default='../Datasets/java/')
parser.add_argument('--log_dir', type=str, default='../XXX')

# Model Architecture
parser.add_argument("--filters", type=str, default='100', help="filter sizes for graph convolution network.")
parser.add_argument("--conv", type=str, default='rgcn', help="(rgcn/cg/nnconv) the kind of graph neural network.")
parser.add_argument("--match", type=str, default='submul', help="(mul/sub/submul) indicating the matching operation.")
parser.add_argument("--match_agg", type=str, default='fc_max', help="(fc_max/max/fc_avg/avg) indicating the aggregation operation.")
parser.add_argument('--margin', type=float, default=0.5, help="the margin value for the ranking loss function.")

# training parameters
parser.add_argument('--max_iter', type=int, default=216259, help="Number of training iterations.")
parser.add_argument('--val_start', type=int, default=100000, help="Number of iterations to start validation.")
parser.add_argument("--train_batch_size", type=int, default=10, help="Number of graph pairs per batch for Training.")
parser.add_argument('--valid_batch_size', type=int, default=50, help="Number of graph pairs per batch for Validation/Testing.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")

# testing
parser.add_argument('--only_test', type=lambda x: (str(x).lower() == 'true'), default='false')
parser.add_argument('--model_path', type=str, default='.')

# others
parser.add_argument('--print_interval', type=int, default=2000)
parser.add_argument('--valid_interval', type=int, default=10000)
parser.add_argument('--gpu_index', type=str, default='3', help="gpu index to use")

args = parser.parse_args()

args_format = 'Filter_{}_1CONV_{}_2MATCH_{}_3MatchAgg_{}_margin_{}_4MaxIter_{}_trainBS_{}_validBS_{}_LR_{}_Dropout_{}'.format(args.filters, args.conv, args.match, args.match_agg,
                                                                                                                              args.margin, args.max_iter, args.train_batch_size,
                                                                                                                              args.valid_batch_size, args.lr, args.dropout)

print(args_format)
