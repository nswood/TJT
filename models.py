import numpy as np
import torch
import torch.nn as nn
import itertools
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Defines the interaction matrices
from torch.nn import Sequential, Linear
from hyp_utils import *

import sys
sys.path.append('HypJets')
from HypJets.hyptorch import nn as hypnn


import torch.nn.functional as F
#from transformers.modeling_bert import ACT2FN, BertEmbeddings, BertSelfAttention, prune_linear_layer, gelu_new
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfAttention, prune_linear_layer#, gelu_new
import tqdm
from transformers.activations import gelu_new
import math
VERBOSE=False
class OskarAttention(BertSelfAttention):
    def __init__(self, config, isSV=False):
        super().__init__(config)
        self.isSV = isSV
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads

        if isSV:
            self.hidden_size = int(config.hidden_size/2)
            self.attention_head_size = int(config.hidden_size/2) // config.num_attention_heads
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
            self.dense = nn.Linear(int(config.hidden_size/2), int(config.hidden_size/2))
            self.LayerNorm = nn.LayerNorm(int(config.hidden_size/2), eps=config.layer_norm_eps)
        else:
            self.hidden_size = config.hidden_size
            self.attention_head_size = config.hidden_size // config.num_attention_heads
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()
        self.attention_band = config.attention_band
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        #print(input_ids) 
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.attention_band is not None:
            query_layer = query_layer.permute(0, 2, 1, 3)
            key_layer = key_layer.permute(0, 2, 1, 3)
            value_layer = value_layer.permute(0, 2, 1, 3)

            attn_band = self.attention_band
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
                remove_from_windowed_attention_mask = (attention_mask != 0)
            query_layer /= math.sqrt(self.attention_head_size)
            query_layer = query_layer.float().contiguous()
            key_layer = key_layer.float().contiguous()
            if False:
                attention_scores = diagonaled_mm_tvm(
                        query_layer, key_layer,
                        attn_band,
                        1, False, 0, False # dilation, is_t1_diag, padding, autoregressive
                    )
            else:
                attention_scores = sliding_chunks_matmul_qk(
                        query_layer, key_layer,
                        attn_band, padding_value=0
                )
            mask_invalid_locations(attention_scores, attn_band, 1, False)
            if attention_mask is not None:
                remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                float_mask = remove_from_windowed_attention_mask.type_as(query_layer).masked_fill(remove_from_windowed_attention_mask, -10000.0)
                float_mask = float_mask.repeat(1, 1, 1, 1) # don't think I need this
                ones = float_mask.new_ones(size=float_mask.size())
                if False:
                    d_mask = diagonaled_mm_tvm(ones, float_mask, attn_band, 1, False, 0, False)
                else:
                    d_mask = sliding_chunks_matmul_qk(ones, float_mask, attn_band, padding_value=0)
                attention_scores += d_mask

            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
            attention_probs = self.dropout(attention_probs)

            value_layer = value_layer.float().contiguous()
            if False:
                context_layer = diagonaled_mm_tvm(attention_probs, value_layer, attn_band, 1, True, 0, False)
            else:
                context_layer = sliding_chunks_matmul_pv(attention_probs, value_layer, attn_band)

        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            #print(query_layer, key_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                
                #Current bug here, attention_mask is not 105xEmbed, only 100xEmbed
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            if VERBOSE:
                # print(attention_probs[0, :8, :8])
                print(torch.max(attention_probs), torch.min(attention_probs))

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3)

        context_layer = context_layer.contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class OskarLayer(nn.Module):
    def __init__(self, config, isSV):
        super().__init__()

        self.config = config
        if isSV:
            self.full_layer_layer_norm = nn.LayerNorm(int(config.hidden_size/2), eps=config.layer_norm_eps)
            self.attention = OskarAttention(config, True)
            self.ffn = nn.Linear(int(config.hidden_size/2), config.intermediate_size)
            self.ffn_output = nn.Linear(config.intermediate_size, int(config.hidden_size/2))
        else:
            self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.attention = OskarAttention(config)
            self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
            self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        try:
            self.activation = ACT2FN[config.hidden_act]
        except KeyError:
            self.activation = config.hidden_act

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class OskarLayerGroup(nn.Module):
    def __init__(self, config, isSV=False):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([OskarLayer(config,isSV) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class OskarTransformer(nn.Module):
    def __init__(self, config,first = False,  isSV=False):
        super().__init__()
        self.config = config
        self.first = first
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.hidden_size
        if isSV:
            self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, int(self.output_hidden_states/2))
        else:
            if first:
                self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
            else:
                self.embedding_hidden_mapping_in = nn.Linear(config.hidden_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([OskarLayerGroup(config,isSV) for _ in range(config.num_hidden_groups)])
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        
        #Problem here with matching
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
            )
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class Transformer(nn.Module):
    def __init__(self, config, name, softmax, sigmoid,sv_branch=False,pretrain = False):
        super().__init__()
        #print(config)
        self.relu = gelu_new #nn.ReLU() 
        self.tanh = nn.Tanh()
        self.gpu_id = 'cuda'
        self.config = config
        self.name = name
        config.output_attentions = False
        config.output_hidden_states = False
        config.num_hidden_groups = 1
        config.inner_group_num = 1
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = "gelu_new"
        self.softmax = softmax #config
        self.sigmoid = sigmoid
        self.input_bn = nn.BatchNorm1d(config.feature_size)
        self.pretrain = pretrain
        self.hybrid = config.hybrid
        if config.hyperbolic:
            self.c = config.c
        
        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        if sv_branch: 
            self.input_bn_sv = nn.BatchNorm1d(config.feature_sv_size)
            
            self.embedder_sv = nn.Linear(config.feature_sv_size, config.embedding_size)
            #Had to change config.n_out_nodes*2 to config.n_out_nodes in first input to first layer
            

            self.final_embedder_sv = nn.ModuleList([
                nn.Linear(int(config.n_out_nodes), int(config.n_out_nodes*2)),
                nn.Linear(int(config.n_out_nodes*2), int(config.n_out_nodes*4)),
                nn.Linear(int(config.n_out_nodes*4), int(config.n_out_nodes*4)),
                nn.Linear(int(config.n_out_nodes*4), int(config.n_out_nodes*2)),
                nn.Linear(int(config.n_out_nodes*2), int(config.n_out_nodes)),
                nn.Linear(int(config.n_out_nodes), int(config.n_out_nodes)),
                nn.Linear(int(config.n_out_nodes), config.nclasses),
                ])
            self.embed_bn_sv = nn.BatchNorm1d(config.embedding_size)
            self.encoders_sv = nn.ModuleList()
            for i in range(config.num_encoders):
                
                if i == 0:
                    print('should be first')
                    self.encoders_sv.append(OskarTransformer(config,True,False))
                else:
                    self.encoders_sv.append(OskarTransformer(config,False,False))
                
            
            
            self.decoders_sv = nn.ModuleList([
                                           nn.Linear(int(config.hidden_size), int(config.hidden_size)),
                                           nn.Linear(int(config.hidden_size), int(config.hidden_size)),
                                           nn.Linear(int(config.hidden_size), int(config.n_out_nodes))
                                           ])
    
    
    
            self.decoder_bn_sv = nn.ModuleList([nn.BatchNorm1d(int(config.hidden_size)) for _ in self.decoders_sv[:-1]])
           
        
        self.final_embedder = nn.ModuleList([
                                            nn.Linear(config.n_out_nodes, int(config.n_out_nodes/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes/2), int(config.n_out_nodes/2)),
                                            nn.Linear(int(config.n_out_nodes/2), int(config.n_out_nodes*4)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes*4), int(config.n_out_nodes*2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes*2), int(config.n_out_nodes/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes/2), config.nclasses),
                                        ])
        self.first_three_modules = self.final_embedder[:3]
        self.remaining_modules = self.final_embedder[3:]
        
        size = config.nparts
        if config.sv:
            size = size + 5
        if config.hyperbolic:
                self.PtoE =hypnn.FromPoincare(config.c)
        if config.replace_mean:  
            self.pre_final = nn.ModuleList([
                                               nn.Linear(int(size), int(size/2)),
                                               nn.Linear(int(size/2), int(size/4)),
                                               nn.Linear(int(size/4), int(1))
                                               ])
            


        self.embed_bn = nn.BatchNorm1d(config.embedding_size)

        self.encoders = nn.ModuleList()
        for i in range(config.num_encoders):
                
            if i == 0:

                self.encoders.append(OskarTransformer(config,True,False))
            else:
                self.encoders.append(OskarTransformer(config,False,False))
        
                           
        self.decoders = nn.ModuleList([
                               nn.Linear(config.hidden_size, config.hidden_size),
                               nn.Linear(config.hidden_size, config.hidden_size),
                               nn.Linear(config.hidden_size, config.n_out_nodes)
                               ])
       
        self.decoder_bn = nn.ModuleList([nn.BatchNorm1d(config.hidden_size) for _ in self.decoders[:-1]])
        #self.pooling = torch.mean()
        self.tests = nn.ModuleList(
                    [
                      nn.Linear(config.feature_size, 1, bias=False),
                      # nn.Linear(config.feature_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, 1)
                    ]
                    )

        self.config = config
        print(self.decoders)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, sv=None,  mask=None, sv_mask=None):
        
        if mask is None:
            mask = torch.ones(x.size()[:-1], device=device)
        if len(mask.shape) == 3:
            attn_mask = mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
        else:
            attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]

        #print(x)
        attn_mask = (1 - attn_mask) * -1e9
        if self.config.mname is not None:
            attn_mask = attn_mask.to(device)
        head_mask = [None] * self.config.num_hidden_layers
	
	# Embed x
        
        x = self.input_bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        h = self.embedder(x)
        
        h = torch.relu(h)
        h = self.embed_bn(h.permute(0, 2, 1)).permute(0, 2, 1)
        
	# If sv not None, embed sv and concat to x
        if sv is not None:
            
            
            if sv_mask is None:
                
                
                x1 = x.shape[1]
                sv1 = sv.shape[1]
                
                #Used trick to get right shape via hstack to fix mask size bug
#                 concat_mask_size = torch.rand(config.embedding_size, (x1+sv1)).size()
#                 Used trick to get right shape via hstack to fix mask size bug
                concat_mask_size = torch.hstack([x[:,:,0],sv[:,:,0]]).size()
#                 print(concat_mask_size)
                
                sv_mask = torch.ones(concat_mask_size, device=device)
                
            if len(sv_mask.shape) == 3:
                attn_sv_mask = sv_mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
            else:
                attn_sv_mask = sv_mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]
            attn_sv_mask = (1 - attn_sv_mask) * -1e9
            head_sv_mask = [None] * self.config.num_hidden_layers

            x = self.input_bn_sv(sv.permute(0, 2, 1)).permute(0, 2, 1)
            
            j = self.embedder_sv(x)
            j = torch.relu(j)
            j = self.embed_bn_sv(j.permute(0, 2, 1)).permute(0, 2, 1)
           
	    #Now j is hte embedded version of sv and we should stack 
            h = torch.cat((h,j),dim=1)
	    
	    #Shape now is 105xembedding size	
           
            #h is now concatented x and sv, so we process normally to final step	
            for e in self.encoders_sv:
            #print(h,attn_mask,head_mask)
            	h = e(h, attn_sv_mask, head_mask)[0]
#             if self.config.hyperbolic:
#                 h = torch.squeeze(h, dim= 1)
#                 h = hypnn.ToPoincare(self.c)(h)
            h = self.decoders_sv[0](h)
            h = self.relu(h)
            h = self.decoder_bn_sv[0](h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.decoders_sv[1](h)
            h = self.relu(h)
            h = self.decoder_bn_sv[1](h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.decoders_sv[2](h)
            #print("before h.shape",h.shape)
#             print(h.size())
            
            #============================================
            
            if self.config.replace_mean:
                h = torch.reshape(h,(h.shape[0],h.shape[2],h.shape[1]))
                h = self.pre_final[0](h)
                h = self.pre_final[1](h)
                h = self.pre_final[2](h)
                h = torch.squeeze(h,dim =2)
                
            else:
                h = torch.mean(h,dim=1)
           
           
            



            for module in self.first_three_modules:
                 h =module(h)
            if self.pretrain:
                return h
            elif self.hybrid: 
                cspace = h
            for module in self.remaining_modules:
                h =module(h)


        else:
	    # shape is now 100 x embedding size 
            
            for e in self.encoders:
            #print(h,attn_mask,head_mask)
            	h = e(h, attn_mask, head_mask)[0]
#             if self.config.hyperbolic:
#                 h = torch.squeeze(h, dim= 1)
#                 h = hypnn.ToPoincare(self.c)(h)
            h = self.decoders[0](h)
            h = self.relu(h)
            h = self.decoder_bn[0](h.permute(0, 2, 1)).permute(0, 2, 1)

            h = self.decoders[1](h)
            h = self.relu(h)
            h = self.decoder_bn[1](h.permute(0, 2, 1)).permute(0, 2, 1)
            
            
            h = self.decoders[2](h)
            if self.config.replace_mean:
                h = torch.reshape(h,(h.shape[0],h.shape[2],h.shape[1]))
                h = self.pre_final[0](h)
                h = self.pre_final[1](h)
                h = self.pre_final[2](h)
                h = torch.squeeze(h,dim =2)
            else:
                h = torch.mean(h,dim=1)
           
            for module in self.first_three_modules:
                 h =module(h)
            if self.pretrain:
                return h
            elif self.hybrid: 
                cspace = h
            for module in self.remaining_modules:
                h =module(h)
            
            
      
        if self.softmax:
            h = nn.Softmax(dim=1)(h)
        if self.sigmoid:
            h = nn.Sigmoid()(h)
        #sys.exit(1)
        if  self.hybrid:
            return h, cspace
        return h

    
class HypeTrans(nn.Module):
    def __init__(self, config, name, softmax, sigmoid,sv_branch=False,pretrain = False):
        super().__init__()
        #print(config)
        self.relu = gelu_new #nn.ReLU() 
        self.tanh = nn.Tanh()
        self.gpu_id = 'cuda'
        self.config = config
        self.name = name
        config.output_attentions = False
        config.output_hidden_states = False
        config.num_hidden_groups = 1
        config.inner_group_num = 1
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = "gelu_new"
        self.softmax = softmax #config
        self.sigmoid = sigmoid
        self.input_bn = nn.BatchNorm1d(config.feature_size)
        self.pretrain = pretrain
        if config.hyperbolic:
            self.c = config.c
        
#         self.embedder = hypnn.HypLinear(config.feature_size, config.embedding_size,c=self.c)
        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        if sv_branch: 
            self.embedder_sv = nn.Linear(config.feature_sv_size, config.embedding_size)
            self.input_bn_sv = nn.BatchNorm1d(config.feature_sv_size)
            
            
            
            

            self.embed_bn_sv = nn.BatchNorm1d(config.embedding_size)
            self.encoders_sv = nn.ModuleList()
            for i in range(config.num_encoders):
                
                if i == 0:
                    print('should be first')
                    self.encoders_sv.append(OskarTransformer(config,True,False))
                else:
                    self.encoders_sv.append(OskarTransformer(config,False,False))
                
            
            
            self.decoders_sv = nn.ModuleList([
                                           nn.Linear(int(config.hidden_size), int(config.hidden_size)),
                                           nn.Linear(int(config.hidden_size), int(config.hidden_size)),
                                           nn.Linear(int(config.hidden_size), int(config.n_out_nodes))
                                           ])
    
    
    
            self.decoder_bn_sv = nn.ModuleList([nn.BatchNorm1d(int(config.hidden_size)) for _ in self.decoders_sv[:-1]])
           
        
        self.final_embedder = nn.ModuleList([
                                            nn.Linear(config.n_out_nodes, int(config.n_out_nodes/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes/2), int(config.n_out_nodes/2)),
                                            nn.Linear(int(config.n_out_nodes/2), int(config.n_out_nodes*4)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes*4), int(config.n_out_nodes*2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes*2), int(config.n_out_nodes/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes/2), config.nclasses),
                                        ])
        self.first_three_modules = self.final_embedder[:3]
        self.remaining_modules = self.final_embedder[3:]
#         self.final_embedder = nn.ModuleList([
#                                             hypnn.HypLinear(config.n_out_nodes, int(config.n_out_nodes),self.c),
#                                             hypnn.HypLinear(int(config.n_out_nodes), int(config.n_out_nodes),self.c),
#                                             hypnn.HypLinear(int(config.n_out_nodes), int(config.n_out_nodes),self.c),
#                                             nn.Linear(int(config.n_out_nodes), int(config.n_out_nodes/2)),
#                                             nn.Linear(int(config.n_out_nodes/2), config.nclasses),
#                                         ])




        size = config.nparts
        if config.sv:
            size = size + 5
     
        self.PtoE =hypnn.FromPoincare(config.c)
        if config.replace_mean:  
            self.pre_final = nn.ModuleList([
                                               hypnn.HypLinear(int(size), int(size/2),self.c),
                                               hypnn.HypLinear(int(size/2), int(size/4),self.c),
                                               hypnn.HypLinear(int(size/4), int(1),self.c)
                                               ])
            


        self.embed_bn = nn.BatchNorm1d(config.embedding_size)

        self.encoders = nn.ModuleList()
        for i in range(config.num_encoders):
                
            if i == 0:

                self.encoders.append(OskarTransformer(config,True,False))
            else:
                self.encoders.append(OskarTransformer(config,False,False))
        self.decoders = nn.ModuleList([
                                   nn.Linear(config.hidden_size, config.hidden_size),
                                   nn.Linear(config.hidden_size, config.hidden_size),
                                   nn.Linear(config.hidden_size, config.n_out_nodes)
                                   ])
                           
          
        self.decoder_bn = nn.ModuleList([nn.BatchNorm1d(config.hidden_size) for _ in self.decoders[:-1]])
        #self.pooling = torch.mean()
       

        self.config = config
        print(self.decoders)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly/work/tier3/jkrupa/cl/deepjet-geometric/examples/cl_v1_train_t0p1_nloss.py different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.001)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, sv=None,  mask=None, sv_mask=None):
#         x = hypnn.ToPoincare(self.c)(x)
        
        if mask is None:
            mask = torch.ones(x.size()[:-1], device=device)
        if len(mask.shape) == 3:
            attn_mask = mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
        else:
            attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]

        #print(x)
        attn_mask = (1 - attn_mask) * -1e9
        if self.config.mname is not None:
            attn_mask = attn_mask.to(device)
        head_mask = [None] * self.config.num_hidden_layers
	
	# Embed x
        
        x = self.input_bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        h = self.embedder(x)
        
        h = torch.relu(h)
        h = self.embed_bn(h.permute(0, 2, 1)).permute(0, 2, 1)
        
	# If sv not None, embed sv and concat to x
        if sv is not None:
#             sv = hypnn.ToPoincare(self.c)(x)
            
            if sv_mask is None:
                
                
                #Used trick to get right shape via hstack to fix mask size bug
                concat_mask_size = torch.hstack([x[:,:,0],sv[:,:,0]]).size()
                
                sv_mask = torch.ones(concat_mask_size, device=device)
                
            if len(sv_mask.shape) == 3:
                attn_sv_mask = sv_mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
            else:
                attn_sv_mask = sv_mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]
            attn_sv_mask = (1 - attn_sv_mask) * -1e9
            head_sv_mask = [None] * self.config.num_hidden_layers

            x = self.input_bn_sv(sv.permute(0, 2, 1)).permute(0, 2, 1)
            
            j = self.embedder_sv(x)
            j = torch.relu(j)
            j = self.embed_bn_sv(j.permute(0, 2, 1)).permute(0, 2, 1)
           
	    #Now j is hte embedded version of sv and we should stack 
            h = torch.cat((h,j),dim=1)
	    
	    #Shape now is 105xembedding size	
            
                
            #h is now concatented x and sv, so we process normally to final step	
            for e in self.encoders_sv:
            	h = e(h, attn_sv_mask, head_mask)[0]

            
            h = self.decoders_sv[0](h)
            h = self.relu(h)
            h = self.decoder_bn_sv[0](h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.decoders_sv[1](h)
            h = self.relu(h)
            h = self.decoder_bn_sv[1](h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.decoders_sv[2](h)

            h = hypnn.ToPoincare(self.c)(h)
            if self.config.replace_mean:
                h = torch.reshape(h,(h.shape[0],h.shape[2],h.shape[1]))
                h = self.pre_final[0](h)
                h = self.pre_final[1](h)
                h = self.pre_final[2](h)
                h = torch.squeeze(h,dim =2)
                
            else:
                h = torch.mean(h,dim=1)
            
           

            h = torch.nn.functional.normalize(h,dim=1)
            h = hypnn.FromPoincare(self.c)(h)
            for module in self.first_three_modules:
                 h =module(h)
            if self.pretrain:
                return h
            elif self.hybrid: 
                cspace = h
            for module in self.remaining_modules:
                h =module(h)

        else:
	    # shape is now 100 x embedding size 
            
            for e in self.encoders:
            	h = e(h, attn_mask, head_mask)[0]
                
            h = self.decoders[0](h)
            h = self.relu(h)
            h = self.decoder_bn[0](h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.decoders[1](h)
            h = self.relu(h)
            h = self.decoder_bn[1](h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.decoders[2](h)
            
            h = hypnn.ToPoincare(self.c)(h)
            
            if self.config.replace_mean:
                h = torch.reshape(h,(h.shape[0],h.shape[2],h.shape[1]))
                h = self.pre_final[0](h)
                h = self.pre_final[1](h)
                h = self.pre_final[2](h)
                h = torch.squeeze(h,dim =2)
            else:
                h = torch.mean(h,dim=1)
                
            h = torch.nn.functional.normalize(h,dim=1)
            
            h = hypnn.FromPoincare(self.c)(h)
            
            for module in self.first_three_modules:
                 h =module(h)
            if self.pretrain:
                return h
            for module in self.remaining_modules:
                h =module(h)
        
       
        if self.softmax:
            h = nn.Softmax(dim=1)(h)
        if self.sigmoid:
            h = nn.Sigmoid()(h)
        #sys.exit(1)
        return h



class GraphNetv2(nn.Module):
    def __init__(self, name, n_constituents, n_targets, params, n_vertices=0, params_v=0, params_e=0, event_branch=False, pv_branch=False,  vv_branch=False, hidden=20, De=20, Do=20, dropout=0.1, softmax=False, sigmoid=False, attention_flag=False, is_oldmodel=False):
        super(GraphNetv2, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.name = name
        self.is_oldmodel = is_oldmodel
        self.Nv = n_vertices		
        self.N = n_constituents
        self.S = params_v
        self.E = params_e
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.De = De
        self.Do = Do
        self.n_targets = n_targets
        self.assign_matrices()
        self.assign_matrices_SV()
        self.event_branch = event_branch
        self.pv_branch = pv_branch
        self.vv_branch = vv_branch
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.relu = nn.ReLU()
        self.attention_flag = attention_flag 
    
        self.batchnorm = nn.BatchNorm1d(params)        
        self.batchnormSV = nn.BatchNorm1d(params_v)
        if self.event_branch:        
            self.batchnormE = nn.BatchNorm1d(params_e)        

        self.fr1 = nn.Conv1d(2*self.P, 4*self.De, kernel_size=1)#.cuda()
        self.fr2 = nn.Conv1d(4*self.De, 2*self.De, kernel_size=1)#.cuda()
        self.fr3 = nn.Conv1d(2*self.De, self.De, kernel_size=1)#.cuda()
        self.fr_batchnorm = nn.BatchNorm1d(self.De,  momentum=0.6)#.cuda()

        if self.event_branch:
            self.fe1    = nn.Linear(self.E,80)#.cuda()
            self.fe2    = nn.Linear(80,40)#.cuda()
            self.fe3    = nn.Linear(40,int(self.De/2))#.cuda()

        if self.pv_branch:
            self.assign_matrices_SV()
            self.fr1_pv = nn.Conv1d(self.S + self.P, self.hidden, kernel_size=1)#.cuda()
            self.fr2_pv = nn.Conv1d(self.hidden, int(self.hidden), kernel_size=1)#.cuda()
            self.fr3_pv = nn.Conv1d(int(self.hidden), self.De, kernel_size=1)#.cuda()

        if self.vv_branch:
            self.assign_matrices_SVSV()

            self.fr1_vv = nn.Conv1d(2 * self.S + self.Dr, self.hidden, kernel_size=1)#.cuda()
            self.fr2_vv = nn.Conv1d(self.hidden, int(self.hidden), kernel_size=1)#.cuda()
            self.fr3_vv = nn.Conv1d(int(self.hidden), self.De, kernel_size=1)#.cuda() 


        if self.pv_branch:
            self.fo1 = nn.Conv1d(self.P + (2 * self.De), 2*self.hidden, kernel_size=1)#.cuda()
            self.fo2 = nn.Conv1d(2*self.hidden, self.hidden, kernel_size=1)#.cuda()
            self.fo3 = nn.Conv1d(self.hidden, self.Do, kernel_size=1)#.cuda()

        else:            
            self.fo1 = nn.Conv1d(self.P + self.De, 2*self.hidden, kernel_size=1)#.cuda()
            self.fo2 = nn.Conv1d(2*self.hidden, self.hidden, kernel_size=1)#.cuda()
            self.fo3 = nn.Conv1d(self.hidden, self.Do, kernel_size=1)#.cuda()
        
        # Attention stuff
        if attention_flag: 
            self.attention = nn.MultiheadAttention(embed_dim=Do, num_heads=int(Do/2), batch_first=True)#.cuda()
            self.layer_norm_1 = nn.LayerNorm(Do)#.cuda()
            self.layer_norm_2 = nn.LayerNorm(Do)#.cuda()
            self.dropout_1 = nn.Dropout(dropout)#.cuda()
            self.dropout_2 = nn.Dropout(dropout)#.cuda()
            self.dropout_3 = nn.Dropout(dropout)#.cuda()
            self.linear_1 = nn.Linear(Do, Do*2)#.cuda()
            self.linear_2 = nn.Linear(Do*2, Do)#.cuda()
            self.linear_3 = nn.Linear(Do*self.N, Do)#.cuda()
        if self.is_oldmodel: 
            self.fc_fixed = nn.Linear(self.Do, self.n_targets)#.cuda()
        else:
            Ninputs = self.Do
            if self.event_branch: Ninputs+=int(self.De/2)
            self.fc_fixed1 = nn.Linear(Ninputs, 5*(self.n_targets))
            self.fc_fixed2 = nn.Linear(5*self.n_targets, 3*self.n_targets)
            self.fc_fixed3 = nn.Linear(3*self.n_targets, self.n_targets)


            
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).to(device)
        self.Rs = (self.Rs).to(device)

    def assign_matrices_SV(self):
        self.Rk = torch.zeros(self.N, self.Nt)
        self.Rv = torch.zeros(self.Nv, self.Nt)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1
        self.Rk = (self.Rk).to(device)
        self.Rv = (self.Rv).to(device)

    def assign_matrices_SVSV(self):
        self.Rl = torch.zeros(self.Nv, self.Ns)
        self.Ru = torch.zeros(self.Nv, self.Ns)
        receiver_sender_list = [i for i in itertools.product(range(self.Nv), range(self.Nv)) if i[0]!=i[1]]
        for i, (l, u) in enumerate(receiver_sender_list):
            self.Rl[l, i] = 1
            self.Ru[u, i] = 1
        self.Rl = (self.Rl)#.cuda()
        self.Ru = (self.Ru)#.cuda()

    def forward(self, x, y=None, e=None):
        ###PF Candidate - PF Candidate###
        x = self.batchnorm(x)
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = self.relu(self.fr1(B))
        B = self.relu(self.fr2(B))
        E = self.relu(self.fr3(B))
        #E = self.fr_batchnorm(E) 
        del B
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E

       
        ####Secondary Vertex - PF Candidate### 
        if self.pv_branch:
            y = self.batchnormSV(y)
            Ork = self.tmul(x, self.Rk)
            Orv = self.tmul(y, self.Rv)
            B = torch.cat([Ork, Orv], 1)
            #assert torch.isfinite(B).all()
            B = self.relu(self.fr1_pv(B))
            #assert torch.isfinite(B).all()
            B = self.relu(self.fr2_pv(B))
            #assert torch.isfinite(B).all()
            E = self.relu(self.fr3_pv(B))
            #assert torch.isfinite(E).all()
            del B
            Ebar_pv = self.tmul(E, torch.transpose(self.Rk, 0, 1).contiguous())
            #assert torch.isfinite(Ebar_pv).all()
 
      

        ####Final output matrix for particles###
        if self.pv_branch:
            C = torch.cat([x, Ebar_pp, Ebar_pv], 1)
            #assert torch.isfinite(C).all()
            del Ebar_pv
        else:
            C = torch.cat([x, Ebar_pp], 1)

        del Ebar_pp; torch.cuda.empty_cache()
        #C = torch.transpose(C, 2, 1).contiguous()
        ### Second MLP ###
        C = self.relu(self.fo1(C))
        #assert torch.isfinite(C).all()
        C = self.relu(self.fo2(C))
        #assert torch.isfinite(C).all()
        O = self.relu(self.fo3(C))
        #assert torch.isfinite(O).all()
        del C
        O = torch.transpose(O, 1, 2).contiguous()
      
          
        #Taking the sum of over each particle/vertex
        if self.attention_flag: 
            O_norm = self.layer_norm_1(O)
            N = O_norm + self.dropout_1(self.attention(O, O, O, need_weights=False)[0])
            del O_norm
            N2 = self.layer_norm_2(N)
            N = N + self.dropout_3(self.linear_2(self.dropout_2(nn.ReLU()(self.linear_1(N2)))))
            del N2
            N = self.linear_3(torch.flatten(N,start_dim=1))
        else: 
            N = torch.sum(O, dim=1)


        

        if self.event_branch:
            e = self.batchnormE(e) 
            e = self.relu(self.fe1(e))
            e = self.relu(self.fe2(e))
            e = self.relu(self.fe3(e))
            N = torch.cat([N, e], 1)

        #assert torch.isfinite(N).all()
        del O
        
        ### Classification MLP ###
        if self.is_oldmodel: 
            N = self.fc_fixed(N)
        else:
            N = self.fc_fixed1(N)
            N = self.fc_fixed2(N)
            N = self.fc_fixed3(N)
        #print("output",N.shape)
        
        if self.softmax:
            N = nn.Softmax(dim=1)(N)
        elif self.sigmoid: 
            N = nn.Sigmoid()(N)
        return N
        del N; torch.cuda.empty_cache()
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        #return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])
        return torch.mm(x.reshape(-1, x_shape[2]), y).reshape(-1, x_shape[1], y_shape[1])

class GraphNetnoSV(nn.Module):
    def __init__(self, name, n_constituents, n_targets, params, hidden, De=5, Do=6, softmax=False):
        super(GraphNetnoSV, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.name = name
        self.Nv = 0 
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.S = 0
        self.n_targets = n_targets
        self.assign_matrices()
        self.softmax = softmax
           
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden)
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2))
        self.fr3 = nn.Linear(int(self.hidden/2), self.De)
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden)
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden/2))
        self.fr3_pv = nn.Linear(int(self.hidden/2), self.De)



       
        self.fo1 = nn.Linear(self.P + self.Dx + (self.De), self.hidden)
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2))
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do)

        #self.fr1 = nn.DataParallel(self.fr1)        
        #self.fr2 = nn.DataParallel(self.fr2)        
        #self.fr3 = nn.DataParallel(self.fr3)
        #self.fr1_pv = nn.DataParallel(self.fr1_pv)        
        #self.fr2_pv = nn.DataParallel(self.fr2_pv)        
        #self.fr3_pv = nn.DataParallel(self.fr3_pv) 
        #self.fo1 = nn.DataParallel(self.fo1) 
        #self.fo2 = nn.DataParallel(self.fo2) 
        #self.fo3 = nn.DataParallel(self.fo3) 

        
        self.fc_fixed1 = nn.Linear(self.Do, 5*(self.n_targets))
        self.fc_fixed2 = nn.Linear(5*self.n_targets, 3*self.n_targets)
        self.fc_fixed3 = nn.Linear(3*self.n_targets, self.n_targets)
            
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).cuda()
        self.Rs = (self.Rs).cuda()
        #print("self.Rr",self.Rr)
        print("self.Rr.shape",self.Rr.shape)
        #print("self.Rs",self.Rs)
        print("self.Rs.shape",self.Rs.shape)
        
    def forward(self, x):
        ###PF Candidate - PF Candidate###
        print("x.shape",x.shape)
        print("self.Rr.shape",self.Rr.shape)
        Orr = self.tmul(x, self.Rr)
        
        print("Orr.shape",Orr.shape)
        Ors = self.tmul(x, self.Rs)
        print("Ors.shape",Ors.shape)
        B = torch.cat([Orr, Ors], 1)
        print("B0.shape",B.shape)
        #print("B0.shape",B.shape)
        del Orr, Ors
        ### First MLP ###
        #B = torch.transpose(B, 1, 2).contiguous()
        #B = B.contiguous()
        print("B1.shape",B.shape)
        #B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        #print("B1p5 shape",B.reshape(-1, 2 * self.P + self.Dr,self.hidden).shape)
        #B = nn.functional.relu(self.fr1(B.reshape(-1, 2 * self.P + self.Dr,self.hidden)))
        B = nn.functional.relu(self.fr1(B))#reshape(-1, 2 * self.P + self.Dr,self.hidden)))
        print("B2.shape",B.shape)
        B = nn.functional.relu(self.fr2(B))
        print("B3.shape",B.shape)
        #E = nn.functional.relu(self.fr3(B).reshape(-1, self.Nr, self.De))
        E = nn.functional.relu(self.fr3(B))#.reshape(-1, self.Nr, self.De))
        #print("E.shape",E.shape)
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        print("E.shape",E.shape)
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        print("Ebar_pp.shape",Ebar_pp.shape)
        del E
        

        ####Final output matrix for particles###
        

        C = torch.cat([x, Ebar_pp], 1)
        #print("C.shape",C.shape)
        del Ebar_pp
        C = torch.transpose(C, 1, 2).contiguous()
        #print("C.shape",C.shape)
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (self.De))))
        #print("C.shape",C.shape)
        C = nn.functional.relu(self.fo2(C))
        #print("C.shape",C.shape)
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        #print("O.shape",O.shape)
        del C

        
        #Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        #print("N.shape",N.shape)
        del O
        
        ### Classification MLP ###

        N = self.fc_fixed1(N)
        N = self.fc_fixed2(N)
        N = self.fc_fixed3(N)
        #print("MLP: N.shape",N.shape)
        
        if self.softmax:
            N = nn.Softmax(dim=1)(N)
        #torch.cuda.empty_cache()
    
        return N
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        #return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])
        return torch.mm(x.reshape(-1, x_shape[2]), y).reshape(-1, x_shape[1], y_shape[1])

class Linear(nn.Module):
    def __init__(self, n_inputs, n_targets):
        super(Linear, self).__init__()
        self.f1 = nn.Linear(n_inputs, n_targets).cuda()
        self.activation = torch.nn.Sigmoid()
    def forward(self, x): 
        x = self.f1(x)
        return(self.activation(x))

class DNN(nn.Module):
    def __init__(self, name, n_inputs, n_targets):
        super(DNN, self).__init__()
        #self.flat = torch.flatten()
        self.name = name
        self.dropout = nn.Dropout(p=0.25)
        self.b0 = nn.BatchNorm1d(n_inputs).cuda()
        self.f0 = nn.Linear(n_inputs, 50).cuda()
        self.f1 = nn.Linear(50, 40).cuda()
        self.f1b = nn.Linear(40, 40).cuda()
        self.b2 = nn.BatchNorm1d(40).cuda()
        self.f2 = nn.Linear(40, 10).cuda()
        self.b3 = nn.BatchNorm1d(10).cuda()
        self.f3 = nn.Linear(10, 5).cuda()
        self.b5 = nn.BatchNorm1d(5).cuda()
        #self.f4 = nn.Linear(50, 10).cuda()
        self.f5 = nn.Linear(5, n_targets).cuda()
        self.activation = torch.nn.ReLU()
        if n_targets == 2 or n_targets == 1:
            self.lastactivation = torch.nn.Sigmoid()
        elif n_targets > 2:
            self.lastactivation = torch.nn.Softmax(dim=1)
        else:
            raise ValueError("I don't understand n_targets "+str(n_targets))
    def forward(self, x): 
        #print("before flat",x.shape)
        #print("before flat",x[0])
        x = torch.flatten(x,start_dim=1)
        #print("after flat",x.shape)
        #print("after flat",x[1])
        x = self.b0(x)
        x = self.activation(self.f0(x))
        x = self.activation(self.f1(x))
        x = self.activation(self.f1b(x))
        x = self.activation(self.f2(x))
        x = self.b3(x)
        x = self.activation(self.f3(x))
        x = self.b5(x)
        x = self.f5(x)
        #return x
        return(self.lastactivation(x))


import numpy as np
import torch
import torch.nn as nn

'''Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.'''


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 sigmoid=False,
                 for_segmentation=False,
                 event_branch=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)
        self.event_branch = event_branch
        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation

        fcs = []
        if self.event_branch:
            self.ec = nn.Sequential(
                nn.BatchNorm1d(27),
                nn.Linear(27,50),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(50,50),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(50,25),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(25,int(fc_params[0][0]/4)),
                nn.ReLU(),
                nn.Dropout(0.1),
            )                                     
        
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
                if self.event_branch:
                    in_chn = in_chn + int(in_chn/4)
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))

        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))

            
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference
        self.sigmoid = sigmoid 
    def forward(self, points, features, mask=None, event_features=None):
#         print('points:\n', points)
#         print('features:\n', features)
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)
        #print("self.event_branch",self.event_branch)
        #print("event_features",event_features)
        if self.event_branch:
            e = self.ec(event_features)
            x = torch.cat((x,e),dim=1)
        output = self.fc(x)
        if self.sigmoid:
            output = torch.sigmoid(output)
        elif self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        #print(torch.where(~torch.isfinite(x),1,0).nonzero())
        return self.conv(x)


class ParticleNetTagger(nn.Module):

    def __init__(self,
                 name,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 sigmoid=False,
                 event_branch=False,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.name = name
        self.event_branch = event_branch
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference,
                              sigmoid=sigmoid,
                              event_branch=event_branch,
        )

    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask, event_features=None):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask
        #print("pf_points/sv_points",pf_points.shape,sv_points.shape) 
        #print("pf_features/sv_features",pf_features.shape, sv_features.shape)
        #print("pf_mask/sv_mask",pf_mask.shape, sv_mask.shape)
        points = torch.cat((pf_points, sv_points), dim=2)
        #print("self.pf_conv(pf_features * pf_mask)",self.pf_conv(pf_features * pf_mask).shape)
        #print("self.sv_conv(sv_features * sv_mask)",self.sv_conv(sv_features * sv_mask).shape)
        #print("self.pf_conv(pf_features * pf_mask) * pf_mask ",(self.pf_conv(pf_features * pf_mask) * pf_mask).shape)
        #print("self.sv_conv(sv_features * sv_mask) * sv_mask ",(self.sv_conv(sv_features * sv_mask) * sv_mask).shape)
        
        #print("pf_features",pf_features.shape,"\npf_mask",pf_mask.shape,"\nsv_features",sv_features.shape,"\nsv_mask",sv_mask.shape,)
        features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        #print("features",features.shape)
        #features = torch.cat((self.pf_conv(pf_features * pf_mask), self.sv_conv(sv_features * sv_mask)), dim=2)
        mask = torch.cat((pf_mask, sv_mask), dim=2)
        #print("mask",mask.shape)
        #print("points",points.shape)
        #print("features",features.shape)
        #print("mask",mask.shape)
        return self.pn(points, features, mask, event_features)
