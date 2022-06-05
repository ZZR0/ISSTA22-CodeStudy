# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

# Copied from transformers.models.bert.modeling_bert.BertPooler
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
        self.pooler=Pooler(self.config.d_model)
        
    def forward(self, code_inputs=None,prev_code_ids=None,code_lengths=None, 
                      nl_inputs=None,prev_nl_ids=None,nl_lengths=None,
                      return_vec=False): 
        bs=code_inputs.shape[0]
        inputs=torch.cat((code_inputs,nl_inputs),0)
        prev_tokens_ids=torch.cat((prev_code_ids,prev_nl_ids),0)
        lengths=torch.cat((code_lengths,nl_lengths),0)-1

        attention_mask=inputs.ne(self.tokenizer.pad_token_id)
        decoder_attention_mask=prev_tokens_ids.ne(self.tokenizer.pad_token_id)
        decoder_attention_mask[:,0] = 1

        outputs=self.encoder(inputs,attention_mask=attention_mask,
                             decoder_input_ids=prev_tokens_ids,
                             decoder_attention_mask=decoder_attention_mask)[0] # 2B * L * D
        outputs=outputs[range(2*bs),lengths,:] # 2B * D
        outputs=self.pooler(outputs) # 2B * D
        code_vec=outputs[:bs]
        nl_vec=outputs[bs:]
        
        if return_vec:
            return code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,code_vec,nl_vec

      
        
 
