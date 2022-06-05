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

    
    def forward(self, code_inputs,prev_code_inputs,nl_inputs,prev_nl_inputs,return_vec=False): 
        bs=code_inputs.shape[0]
        input_ids=torch.cat((code_inputs,nl_inputs),0)
        prev_tokens_ids=torch.cat((prev_code_inputs,prev_nl_inputs),0)
        lengths = torch.ne(input_ids, 0).sum(-1) - 1
        outputs=self.encoder(src_tokens=input_ids,src_lengths=lengths,prev_output_tokens=prev_tokens_ids,features_only=True)[0] # L * B * D 
        outputs=outputs[range(2*bs),lengths,:] # 3B * D
        code_vec=outputs[:bs]
        nl_vec=outputs[bs:]
        
        if return_vec:
            return code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,code_vec,nl_vec

      
        
 
