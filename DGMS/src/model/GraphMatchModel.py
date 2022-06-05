#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2020/1/1 12:21
# @Author   : Xiang Ling
# @File     : GraphMatchModel.py
# @Lab      : nesa.zju.edu.cn
# ************************************

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.utils import to_dense_batch


class GraphMatchNetwork(torch.nn.Module):
    def __init__(self, node_init_dims, arguments, max_number_of_edges, device):
        super(GraphMatchNetwork, self).__init__()
        
        self.node_init_dims = node_init_dims
        self.args = arguments
        self.device = device
        self.dropout = arguments.dropout
        
        # ---------- Node Embedding Layer ----------
        filters = self.args.filters.split('_')
        self.gcn_filters = [int(n_filter) for n_filter in filters]  # GCNs' filter sizes
        self.gcn_numbers = len(self.gcn_filters)
        self.gcn_last_filter = self.gcn_filters[-1]  # last filter size of node embedding layer
        
        self.max_relations = max_number_of_edges  # number of relations
        
        rgcn_parameters = [dict(in_channels=self.gcn_filters[i - 1], out_channels=self.gcn_filters[i], num_relations=self.max_relations, num_bases=10, bias=True)
                           for i in range(1, self.gcn_numbers)]
        rgcn_parameters.insert(0, dict(in_channels=node_init_dims, out_channels=self.gcn_filters[0], num_relations=self.max_relations, num_bases=10, bias=True))
        
        conv_layer_constructor = {
            'rgcn': dict(constructor=RGCNConv, kwargs=rgcn_parameters)
        }
        
        conv = conv_layer_constructor[self.args.conv]
        constructor = conv['constructor']
        # build Graph Encoding layers
        setattr(self, 'GraphEncoder_{}'.format(1), constructor(**conv['kwargs'][0]))
        for i in range(1, self.gcn_numbers):
            setattr(self, 'GraphEncoder_{}'.format(i + 1), constructor(**conv['kwargs'][i]))
        
        # ---------- Semantic Matching Layer ----------
        if self.args.match.lower() == 'sub' or self.args.match.lower() == 'mul':
            self.agg_input_size = self.gcn_last_filter
        elif self.args.match.lower() == 'submul':
            self.agg_input_size = self.gcn_last_filter
            self.fc_match = nn.Linear(in_features=self.gcn_last_filter * 2, out_features=self.gcn_last_filter)
        
        else:
            raise NotImplementedError
        
        # ---------- Aggregation Layer ----------
        if self.args.match_agg.lower() == 'fc_avg' or self.args.match_agg.lower() == 'fc_max':
            self.fc_agg = nn.Linear(self.agg_input_size, self.agg_input_size)
        elif self.args.match_agg.lower() == 'avg' or self.args.match_agg.lower() == 'max':
            pass
        else:
            raise NotImplementedError
    
    @staticmethod
    def div_with_small_value(n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d
    
    def cosine_attention(self, v1, v2):
        """
        :param v1: (batch, len1, dim)
        :param v2: (batch, len2, dim)
        :return:  (batch, len1, len2)
        """
        # (batch, len1, len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1))
        
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)
    
    def forward_message_passing_layers(self, x, edge_index, edge_attr):
        x_in = x
        for i in range(1, self.gcn_numbers + 1):
            x_out = functional.relu(getattr(self, 'GraphEncoder_{}'.format(i))(x_in, edge_index, edge_attr), inplace=True)
            x_in = x_out
        return x_out
    
    def forward(self, batch_1, batch_2):
        
        # ---------- Node Embedding Layer ----------
        batch_1 = batch_1.to(self.device)
        batch_2 = batch_2.to(self.device)
        
        if self.args.conv == 'rgcn':
            b1_edge_type = torch.argmax(batch_1.edge_attr, dim=1)
            b2_edge_type = torch.argmax(batch_2.edge_attr, dim=1)
            feature_p = self.forward_message_passing_layers(x=batch_1.x, edge_index=batch_1.edge_index, edge_attr=b1_edge_type)
            feature_h = self.forward_message_passing_layers(x=batch_2.x, edge_index=batch_2.edge_index, edge_attr=b2_edge_type)
        else:
            feature_p = self.forward_message_passing_layers(x=batch_1.x, edge_index=batch_1.edge_index, edge_attr=batch_1.edge_attr)
            feature_h = self.forward_message_passing_layers(x=batch_2.x, edge_index=batch_2.edge_index, edge_attr=batch_2.edge_attr)
        
        feature_p = to_dense_batch(x=feature_p, batch=batch_1.batch)[0]
        feature_h = to_dense_batch(x=feature_h, batch=batch_2.batch)[0]
        
        # ---------- Semantic Matching Layer ----------
        
        attention = self.cosine_attention(feature_p, feature_h)
        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(3)
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(3)
        
        att_mean_h = self.div_with_small_value(attention_h.sum(dim=2), attention.sum(dim=2, keepdim=True))
        att_mean_p = self.div_with_small_value(attention_p.sum(dim=1), attention.sum(dim=1, keepdim=True).permute(0, 2, 1))
        
        if self.args.match.lower() == 'sub':
            multi_p = (feature_p - att_mean_h) * (feature_p - att_mean_h)
            multi_h = (feature_h - att_mean_p) * (feature_h - att_mean_p)
        elif self.args.match.lower() == 'mul':
            multi_p = feature_p * att_mean_h
            multi_h = feature_h * att_mean_p
        elif self.args.match.lower() == 'submul':
            multi_p = functional.relu(self.fc_match(torch.cat(((feature_p - att_mean_h) * (feature_p - att_mean_h), feature_p * att_mean_h), dim=-1)))
            multi_h = functional.relu(self.fc_match(torch.cat(((feature_h - att_mean_p) * (feature_h - att_mean_p), feature_h * att_mean_p), dim=-1)))
        else:
            raise NotImplementedError
        
        match_p = multi_p
        match_h = multi_h
        
        # ---------- Aggregation Layer ----------
        if self.args.match_agg.lower() == 'avg':
            agg_p = torch.mean(match_p, dim=1)
            agg_h = torch.mean(match_h, dim=1)
        elif self.args.match_agg.lower() == 'fc_avg':
            agg_p = torch.mean(self.fc_agg(match_p), dim=1)
            agg_h = torch.mean(self.fc_agg(match_h), dim=1)
        elif self.args.match_agg.lower() == 'max':
            agg_p = torch.max(match_p, dim=1)[0]
            agg_h = torch.max(match_h, dim=1)[0]
        elif self.args.match_agg.lower() == 'fc_max':
            agg_p = torch.max(self.fc_agg(match_p), dim=1)[0]
            agg_h = torch.max(self.fc_agg(match_h), dim=1)[0]
        else:
            raise NotImplementedError
        
        # ---------- Prediction Layer ----------
        sim = functional.cosine_similarity(agg_p, agg_h, dim=1).clamp(min=-1, max=1)
        return sim
