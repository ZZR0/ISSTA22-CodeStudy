#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/12/23 14:15
# @Author   : Xiang Ling
# @File     : train.py
# @Lab      : nesa.zju.edu.cn
# ************************************


import os
from datetime import datetime

import numpy as np
import torch

from ProcessedDataset import ProcessedDataset
from config import args as arguments
from config import args_format as args_file_name
from model.GraphMatchModel import GraphMatchNetwork
from utils import write_log_file, arguments_to_tables, chunk

os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu_index)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.data_dir
        
        if self.args.only_test:
            self.sig = os.path.join(args.log_dir, "OnlyText_" + datetime.now().strftime("%Y-%m-%d@%H:%M:%S"))
        else:
            self.sig = os.path.join(args.log_dir, datetime.now().strftime("%Y-%m-%d@%H:%M:%S"))
        os.mkdir(self.sig)
        
        self.log_path = os.path.join(self.sig, 'log_{}.txt'.format(args_file_name))
        self.best_model_path = os.path.join(self.sig, 'best_model.pt')
        
        table_draw = arguments_to_tables(args=arguments)
        write_log_file(self.log_path, str(table_draw))
        
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.max_iteration = args.max_iter
        self.margin = args.margin
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        write_log_file(self.log_path, "\n****CPU or GPU: " + str(self.device))
        
        max_number_edge_types = 3
        
        if self.args.conv.lower() in ['rgcn', 'cg', 'nnconv']:
            self.model = GraphMatchNetwork(node_init_dims=300, arguments=args, device=self.device, max_number_of_edges=max_number_edge_types).to(self.device)
        else:
            raise NotImplementedError
        
        write_log_file(self.log_path, str(self.model))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
        write_log_file(self.log_path, "Init Reading Code Graphs ... ")
        self.code_data = ProcessedDataset(name='code', root=self.dataset_dir, log_path=self.log_path)
        write_log_file(self.log_path, "Init Reading Text Graphs ... ")
        self.text_data = ProcessedDataset(name='text', root=self.dataset_dir, log_path=self.log_path)
        
        # for plotting and record (init empty list)
        self.train_iter, self.train_smooth_loss, self.valid_iter, self.valid_loss, self.test_iter, self.test_mrr, self.test_s1, self.test_s5, self.test_s10 = ([] for _ in range(9))
    
    def fit(self):
        best_val_loss = 1e10
        all_loss = []
        code_train_batch = self.code_data.triple_train_batch(self.train_batch_size)
        time_1 = datetime.now()
        for iteration in range(self.max_iteration):
            self.model.train()
            # Compute similarity
            pos_code_graph_id_list, text_graph_id_list, neg_code_graph_id_list = next(code_train_batch)  # next for yield
            pos_code_batch = self.code_data.get_batch_graph(pos_code_graph_id_list)
            text_batch = self.text_data.get_batch_graph(text_graph_id_list)
            neg_code_batch = self.code_data.get_batch_graph(neg_code_graph_id_list)
            
            pos_pred = self.model(pos_code_batch, text_batch).reshape(-1, 1)  # [batch, 1]
            neg_pred = self.model(neg_code_batch, text_batch).reshape(-1, 1)
            
            loss = (self.margin - pos_pred + neg_pred).clamp(min=1e-6).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_loss.append(loss)
            # Print
            if iteration % self.args.print_interval == 0 and iteration > 0:
                self.train_iter.append(iteration)
                self.train_smooth_loss.append(torch.tensor(all_loss).mean().cpu().detach())
                
                write_log_file(self.log_path, '@Train Iter {}: mean smooth loss = @{}@, time = {}.'.format(iteration, torch.tensor(all_loss).mean(), datetime.now() - time_1))
                all_loss = []
                time_1 = datetime.now()
            # Validation
            if (iteration % self.args.valid_interval == 0 and iteration >= self.args.val_start) or iteration == 0:
                s_time = datetime.now()
                loss = self.validation()
                self.valid_iter.append(iteration)
                self.valid_loss.append(loss.cpu().detach())
                end_time = datetime.now()
                if loss < best_val_loss:
                    write_log_file(self.log_path, '#Valid Iter {}: loss = #{}# (Decrease) < Best loss = {}. Save to best model..., time elapsed = {}.'.format(iteration, loss,
                                                                                                                                                              best_val_loss,
                                                                                                                                                              end_time - s_time))
                    best_val_loss = loss
                    torch.save(self.model.state_dict(), self.best_model_path)
                else:
                    write_log_file(self.log_path, '#Valid Iter {}: loss = #{}# (Increase). Best val loss = {}, time elapsed = {}.'.format(iteration, loss, best_val_loss,
                                                                                                                                          end_time - s_time))
            # only testing when iteration == 0 (whether code is rightly run)
            if iteration == 0:
                self.test(iter_no=iteration)
    
    def validation(self):
        """
        Perform a validation using code as base data.
        :return: mean validation loss over the whole validation set.
        """
        with torch.no_grad():
            self.model.eval()
            val_loss = []
            for pos_code_gid_list, text_gid_list, neg_code_gid_list in self.code_data.triple_valid_batch(self.valid_batch_size):
                pos_code_batch = self.code_data.get_batch_graph(pos_code_gid_list)
                text_batch = self.text_data.get_batch_graph(text_gid_list)
                neg_code_batch = self.code_data.get_batch_graph(neg_code_gid_list)
                
                pos_pred = self.model(pos_code_batch, text_batch).reshape(-1, 1)
                neg_pred = self.model(neg_code_batch, text_batch).reshape(-1, 1)
                loss = (self.margin - pos_pred + neg_pred).clamp(min=1e-6).mean()
                
                val_loss.append(loss)
            loss = torch.tensor(val_loss).mean()
        return loss
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def retrieve_rank(self, query_id, candidate_id_list, query_data, cand_data):
        st = 0
        rank = dict()
        one_query_scores = []
        while st < len(candidate_id_list):
            ed = st + self.valid_batch_size if st + self.valid_batch_size < len(candidate_id_list) else len(candidate_id_list)
            code_graph_list, text_graph_list = [], []
            for i in range(st, ed):
                code_graph_list.append(query_id)
                text_graph_list.append(candidate_id_list[i])
            code_batch = query_data.get_batch_graph(code_graph_list)
            text_batch = cand_data.get_batch_graph(text_graph_list)
            self.model.eval()
            with torch.no_grad():
                score = self.model(code_batch, text_batch)
            for candidate_id in text_graph_list:
                rank[candidate_id] = score[text_graph_list.index(candidate_id)]
            one_query_scores.extend(score.tolist())
            st = ed
        
        rank = [a[0] for a in sorted(list(rank.items()), key=lambda x: x[1], reverse=True)]
        assert len(one_query_scores) == len(candidate_id_list), "must be equal, ERROR"
        return rank, np.array(one_query_scores)
    
    @staticmethod
    def calculate_square_mrr(similarity):
        assert similarity.shape[0] == similarity.shape[1]
        correct_scores = np.diagonal(similarity)
        compared_scores = similarity >= correct_scores[..., np.newaxis]
        rrs = 1.0 / compared_scores.astype(np.float).sum(-1)
        return rrs
    
    def test(self, iter_no):
        write_log_file(self.log_path, "Start to testing ...")
        test_query_ids = self.text_data.split_ids['test']
        success = {1: 0, 5: 0, 10: 0}
        total_test_scores = []
        test_start = datetime.now()
        for test_chunk in chunk(test_query_ids, 100):
            one_chunk_scores = []
            for i, query_id in enumerate(test_chunk):
                rank_ids, one_row_scores = self.retrieve_rank(query_id, test_chunk, self.text_data, self.code_data)
                one_chunk_scores.append(one_row_scores)
                for k in success.keys():
                    if query_id in rank_ids[:k]:
                        success[k] += 1
            total_test_scores.append(one_chunk_scores)
        
        write_log_file(self.log_path, "\n&Testing Iteration {}: for {} queries finished. Time elapsed = {}.".format(iter_no, len(test_query_ids), datetime.now() - test_start))
        
        all_mrr = []
        for i in range(len(total_test_scores)):
            one_chunk_square_score = total_test_scores[i]
            one_chunk_square_score = np.vstack(one_chunk_square_score)
            assert one_chunk_square_score.shape[0] == one_chunk_square_score.shape[1], "Every Chunk must be square"
            mrr_array = self.calculate_square_mrr(one_chunk_square_score)
            all_mrr.extend(mrr_array)
        mrr = np.array(all_mrr).mean()
        self.test_iter.append(iter_no)
        self.test_mrr.append(mrr)
        write_log_file(self.log_path, "&Testing Iteration {}: MRR = &{}&".format(iter_no, mrr))
        
        for k, v in success.items():
            value = v * 1.0 / len(test_query_ids)
            write_log_file(self.log_path, "&Testing Iteration {}: S@{}@ = &{}&".format(iter_no, k, value))
            if k == 1:
                self.test_s1.append(value)
            elif k == 5:
                self.test_s5.append(value)
            elif k == 10:
                self.test_s10.append(value)
            else:
                print('cannot find !')
        write_log_file(self.log_path, "S@1, S@5, S@10\n{}, {}, {}".format(self.test_s1[-1], self.test_s5[-1], self.test_s10[-1]))


if __name__ == '__main__':
    all_time_1 = datetime.now()
    trainer = Trainer(arguments)
    if arguments.only_test:
        trainer.load_model(arguments.model_path)
    else:
        trainer.fit()
        trainer.load_model(trainer.best_model_path)
    
    all_time_1 = datetime.now()
    write_log_file(trainer.log_path, "finished to load the model, next to start to test and time is = {}".format(all_time_1))
    trainer.test(iter_no=trainer.max_iteration + 1)
    write_log_file(trainer.log_path, "\nAll Finished using ({})\n".format(datetime.now() - all_time_1))
