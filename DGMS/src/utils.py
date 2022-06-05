#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/12/22 19:55
# @Author   : Xiang Ling
# @File     : utils.py
# @Lab      : nesa.zju.edu.cn
# ************************************
import os

from texttable import Texttable


def chunk(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def write_log_file(file_name_path, log_str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')


def arguments_to_tables(args):
    """
    util function to print the logs in table format
    :param args: parameters
    :return:
    """
    args = vars(args)
    keys = sorted(args.keys())
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_precision(width=10)
    table.set_cols_dtype(['t', 't'])
    table.set_cols_align(['l', 'l'])
    table.add_rows([["Parameter", "Value"]])
    for k in keys:
        table.add_row([k, args[k]])
    return table.draw()


def int_2_one_hot(n, n_classes):
    v = [0] * n_classes
    v[n] = 1
    return v
