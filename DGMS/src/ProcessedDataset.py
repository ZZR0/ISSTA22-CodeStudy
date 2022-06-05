#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2020/1/3 0:36
# @Author   : Xiang Ling
# @File     : ProcessedDataset.py
# @Lab      : nesa.zju.edu.cn
# ************************************
import json
import os
import pickle
import random
import torch
from datetime import datetime
from torch_geometric.data import Batch
from tqdm import tqdm

from utils import write_log_file


class ProcessedDataset(object):
    def __init__(self, name, root, log_path):
        self.name = name
        self.data_processed_path = os.path.join(root, '{}_processed'.format(self.name))
        self.graph_id_file = os.path.join(root, '{}_graph_ids.pt'.format(self.name))
        self.graph_id_list = torch.load(self.graph_id_file)
        self.log_path = log_path
        
        self._check_whether_all_graph_ids_files_exist()
        self.total_graph = {}
        # self.get_total_graphs()
        
        # Split train, test, validation set
        if os.path.exists(os.path.join(root, 'split.json')):
            with open(os.path.join(root, 'split.json'), 'rb') as f:
                self.split_ids = json.loads(f.read())
        else:
            raise NotImplementedError
        write_log_file(self.log_path, "Train={}\nValid={}\nTest={}".format(len(self.split_ids['train']), len(self.split_ids['valid']), len(self.split_ids['test'])))
    
    def _check_whether_all_graph_ids_files_exist(self):
        graph_ids_files = self.processed_file_names()
        for id_file in graph_ids_files:
            if os.path.isfile(id_file) is False:
                raise FileNotFoundError
    
    def processed_file_names(self):
        return [os.path.join(self.data_processed_path, '{}_{}.pt'.format(self.name, ids)) for ids in self.graph_id_list]
    
    def get_total_graphs(self):
        time_1 = datetime.now()
        
        for graph_idx in tqdm(self.graph_id_list):
            data = torch.load(os.path.join(self.data_processed_path, '{}_{}.pt'.format(self.name, graph_idx)))
            self.total_graph[graph_idx] = data
        write_log_file(self.log_path, "load and append {} graph, time = {}".format(len(self.graph_id_list), datetime.now() - time_1))
    
    def get_one_graph(self, idx):
        return torch.load(os.path.join(self.data_processed_path, '{}_{}.pt'.format(self.name, idx)))
    
    def get_batch_graph(self, gid_list):
        """
        Convert a list of graph ids to a PyG mini-batch.
        :param gid_list: list of graph ids
        :return: PyG Batch consisting of corresponding graphs
        """
        batch = []
        for gid in gid_list:
            batch.append(self.get_one_graph(idx=gid))
        return Batch().from_data_list(batch)
    
    def triple_train_batch(self, batch_size):
        train_graph_ids = self.split_ids['train']
        st = 0
        current = 0
        while True:
            ed = st + batch_size if st + batch_size < len(train_graph_ids) else len(train_graph_ids)
            pos_code_list, text_list, neg_code_list = [], [], []
            
            if current >= len(train_graph_ids):  # shuffle the train_graph_ids when finish one epoch
                current = 0
                random.Random(123).shuffle(train_graph_ids)
            
            for i in range(st, ed):
                # positive pair
                pos_code_list.append(train_graph_ids[i])
                text_list.append(train_graph_ids[i])
                # negative
                neg_id_index = random.randint(a=0, b=len(train_graph_ids) - 1)
                while train_graph_ids[neg_id_index] == train_graph_ids[i]:
                    neg_id_index = random.randint(a=0, b=len(train_graph_ids) - 1)
                neg_code_list.append(train_graph_ids[neg_id_index])
            yield pos_code_list, text_list, neg_code_list
            st = ed if ed < len(train_graph_ids) else 0
            current += batch_size
    
    def triple_valid_batch(self, batch_size):
        valid_graph_ids = self.split_ids['valid']
        st = 0
        while st < len(valid_graph_ids):
            ed = st + batch_size if st + batch_size < len(valid_graph_ids) else len(valid_graph_ids)
            pos_code_list, text_list, neg_code_list = [], [], []
            for i in range(st, ed):
                # Generate a positive pair
                pos_code_list.append(valid_graph_ids[i])
                text_list.append(valid_graph_ids[i])
                # negative
                neg_id_index = random.randint(a=0, b=len(valid_graph_ids) - 1)
                while valid_graph_ids[neg_id_index] == valid_graph_ids[i]:
                    neg_id_index = random.randint(a=0, b=len(valid_graph_ids) - 1)
                neg_code_list.append(valid_graph_ids[neg_id_index])
            yield pos_code_list, text_list, neg_code_list
            st = ed
