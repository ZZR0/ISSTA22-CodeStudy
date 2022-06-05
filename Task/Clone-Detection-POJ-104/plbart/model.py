# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.pooler = Pooler(self.config.encoder_embed_dim)
    
        
    def forward(self, input_ids=None,prev_tokens_ids=None,lengths=None,
                      p_input_ids=None,p_prev_tokens_ids=None,p_lengths=None,
                      n_input_ids=None,n_prev_tokens_ids=None,n_lengths=None,
                      labels=None): 
        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids,n_input_ids),0)
        prev_tokens_ids=torch.cat((prev_tokens_ids,p_prev_tokens_ids,n_prev_tokens_ids),0)
        lengths=torch.cat((lengths,p_lengths,n_lengths),0)-1
        
        outputs=self.encoder(src_tokens=input_ids,src_lengths=lengths,prev_output_tokens=prev_tokens_ids,features_only=True)[0] # L * 3B * D
        outputs=outputs[range(3*bs),lengths,:] # 3B * D
        outputs=self.pooler(outputs) # 3B * D
        outputs=outputs.split(bs,0) # B * D , B * D , B * D 
        
        prob_1=(outputs[0]*outputs[1]).sum(-1) # B
        prob_2=(outputs[0]*outputs[2]).sum(-1)
        temp=torch.cat((outputs[0],outputs[1]),0) # 2B * D
        temp_labels=torch.cat((labels,labels),0) # 2B
        prob_3= torch.mm(outputs[0],temp.t()) # B * 2B
        mask=labels[:,None]==temp_labels[None,:] # B * 2B
        prob_3=prob_3*(1-mask.float())-1e9*mask.float() # B * 2B
        
        prob=torch.softmax(torch.cat((prob_1[:,None],prob_2[:,None],prob_3),-1),-1) # # B * 2B+2
        loss=torch.log(prob[:,0]+1e-10)
        loss=-loss.mean()
        return loss,outputs[0]

      
        
 
