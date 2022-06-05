# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,position_idx=None,attn_mask=None,
                p_input_ids=None,p_position_idx=None,p_attn_mask=None,
                n_input_ids=None,n_position_idx=None,n_attn_mask=None,
                labels=None): 

        bs,_=input_ids.size()
        input_ids=torch.cat((input_ids,p_input_ids,n_input_ids),0)
        position_idx=torch.cat((position_idx,p_position_idx,n_position_idx),0)
        attn_mask=torch.cat((attn_mask,p_attn_mask,n_attn_mask),0)
        
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(input_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask # None to add one dimention
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None] # normalization
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    

        # inputs_embeddings: 96 * 640 * 768 
        # attn_mask: 96 * 640 * 640 
        # position_idx: 96 * 640
        outputs=self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1] # 3B * D
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
