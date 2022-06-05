# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from representjs.models.encoder import CodeEncoder

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

class CloneDetectionEncoder(nn.Module):
    def __init__(
        self,
        n_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        encoder_type="transformer",
    ):
        super(CloneDetectionEncoder, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.d_model = d_model
        # Encoder and output for type prediction
        assert encoder_type in ["transformer", "lstm"]
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(
                n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id, project=False
            )
        self.pooler = Pooler(d_model)

    def forward(self, src_tok_ids, lengths=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """

        # Encode
        memory, _ = self.encoder(src_tok_ids, lengths)  # LxBxD
        x = memory[0, :, :] # BxD
        x = self.pooler(x)

        return (memory, x)
    
class Model(nn.Module):   
    def __init__(self, encoder,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,p_input_ids=None,n_input_ids=None,labels=None): 
        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids,n_input_ids),0)
        
        outputs=self.encoder(input_ids)[1] # 3B * D
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

      
        
 
