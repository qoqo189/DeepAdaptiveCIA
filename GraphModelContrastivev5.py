import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
import math


class SemanticFilter(nn.Module):
    def __init__(self,
                 all_embs_list:list,
                 splitlines:torch.Tensor,
                 hidden_size:int=768,
                 device:str="cuda",
                 ):
        super().__init__()
        self.hidden_size=hidden_size
        self.query=nn.Linear(hidden_size,1)
        self.all_embs_list=all_embs_list
        self.splitlines:torch.Tensor=splitlines
        self.device=device


    def forward(self,
                inds:list,
                infonce_temperature:float=0.1,
                device:str="cuda",
                node_predict_indexs=None,
                node_predict_labels=None,
                node_predict_types=None,
                change_node_indexs=None
                ):
        embs_list=[]
        for i in inds:
            e = torch.stack(self.all_embs_list[self.splitlines[i][0].item():self.splitlines[i][1].item()]).squeeze(1)
            e.to(device)
            embs_list.append(e)
        node_embeddings = self.merge(embs_list)

        target_embeddings=node_embeddings[node_predict_indexs]
        source_embeddings=node_embeddings[change_node_indexs]

        info_loss,similarities=self.loss_fn_infonce(target_embeddings,source_embeddings,node_predict_types,
                                                    node_predict_labels,infonce_temperature)

        return info_loss
        
    
    def merge(self, embs_list):
        lengths = [e.size(0) for e in embs_list]
        padded = pad_sequence(embs_list, batch_first=True).to(self.device)

        scores = self.query(padded).squeeze(-1)

        B, N_max = scores.size()
        device = scores.device
        mask = torch.arange(N_max, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)

        scores_masked = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores_masked, dim=1)

        output = torch.sum(padded * weights.unsqueeze(-1), dim=1)

        return output
    

    def loss_fn_infonce(self,target_embeddings:torch.Tensor,
                        source_embeddings:torch.Tensor,
                        node_predict_types:torch.Tensor,
                        node_predict_labels:torch.Tensor,
                        infonce_temperature:float):
        similarities = torch.zeros(target_embeddings.shape[0], 1, 
                                dtype=torch.float, device=target_embeddings.device)
        total_loss = 0.0
        num_valid_queries = 0
        
        for i, cur_type in enumerate(torch.unique(node_predict_types)):
            cur_indices = torch.where(node_predict_types == cur_type)[0]
            cur_source = source_embeddings[cur_type]
            cur_targets = target_embeddings[cur_indices]
            
            expanded_source = cur_source.unsqueeze(0).expand_as(cur_targets)
            concat_input = torch.cat([expanded_source, cur_targets], dim=1)
            cur_similarities = self.w_metric(concat_input)
            similarities[cur_indices] = cur_similarities
            
            cur_labels = node_predict_labels[cur_indices].squeeze(1)
            pos_mask = (cur_labels == 1)
            
            if not torch.any(pos_mask):
                continue
            
            logits = cur_similarities.squeeze(1) / infonce_temperature
            log_denominator = torch.logsumexp(logits, dim=0)
            log_numerator = torch.logsumexp(logits[pos_mask], dim=0)
            
            query_loss = log_denominator - log_numerator
            total_loss += query_loss
            num_valid_queries += 1
        
        if num_valid_queries > 0:
            total_loss = total_loss / num_valid_queries
        else:
            total_loss = torch.tensor(0.0, device=target_embeddings.device)

        return total_loss, similarities
    

    def initialize_model(self):
        print("init semantic filter parameters.\n")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)


class KFilter(nn.Module):
    def __init__(self,h_token:int=768) -> None:
        super().__init__()
        self.w1_bi40=nn.Linear(h_token,384)
        self.w2_bi40=nn.Linear(384,1)

        self.norm=nn.LayerNorm(h_token)

    
    def forward_bi(self,
                   target_embeddings:torch.Tensor,
                   source_embeddings:torch.Tensor,
                   k:int):
        expanded_source = source_embeddings.unsqueeze(0).expand_as(target_embeddings)
        ts_pairs = torch.cat([expanded_source, target_embeddings], dim=1)
        ts_pairs=self.norm(ts_pairs)

        ts_pairs=self.w1_bi40(ts_pairs)
        ts_pairs=F.dropout(F.leaky_relu(ts_pairs),p=0.1,training=self.training)
        bi_logits=self.w2_bi40(ts_pairs)
        
        return bi_logits.squeeze(-1)
    

    def initialize_model(self):
        print("init filter parameters.\n")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)


class KNFilter(nn.Module):
    def __init__(self,h_token:int=768) -> None:
        super().__init__()
        self.w1_bi5=nn.Linear(h_token,384)
        self.w2_bi5=nn.Linear(384,1)
        
        self.w1_bi10=nn.Linear(h_token,384)
        self.w2_bi10=nn.Linear(384,1)

        self.w1_bi20=nn.Linear(h_token,384)
        self.w2_bi20=nn.Linear(384,1)

        self.w1_bi30=nn.Linear(h_token,384)
        self.w2_bi30=nn.Linear(384,1)

        self.w1_bi40=nn.Linear(h_token,384)
        self.w2_bi40=nn.Linear(384,1)

        self.norm=nn.LayerNorm(h_token)

    
    def forward_bi(self,
                   target_embeddings:torch.Tensor,
                   source_embeddings:torch.Tensor,
                   k:int,
                   ):
        expanded_source = source_embeddings.unsqueeze(0).expand_as(target_embeddings)
        ts_pairs = torch.cat([expanded_source, target_embeddings], dim=1)
        ts_pairs=self.norm(ts_pairs)

        if k==5:
            ts_pairs=self.w1_bi5(ts_pairs)
            ts_pairs=F.dropout(F.leaky_relu(ts_pairs),p=0.1,training=self.training)
            bi_logits=self.w2_bi5(ts_pairs)
        elif k==10:
            ts_pairs=self.w1_bi10(ts_pairs)
            ts_pairs=F.dropout(F.leaky_relu(ts_pairs),p=0.1,training=self.training)
            bi_logits=self.w2_bi10(ts_pairs)
        elif k==20:
            ts_pairs=self.w1_bi20(ts_pairs)
            ts_pairs=F.dropout(F.leaky_relu(ts_pairs),p=0.1,training=self.training)
            bi_logits=self.w2_bi20(ts_pairs)
        elif k==30:
            ts_pairs=self.w1_bi30(ts_pairs)
            ts_pairs=F.dropout(F.leaky_relu(ts_pairs),p=0.1,training=self.training)
            bi_logits=self.w2_bi30(ts_pairs)
        elif k==40:
            ts_pairs=self.w1_bi40(ts_pairs)
            ts_pairs=F.dropout(F.leaky_relu(ts_pairs),p=0.1,training=self.training)
            bi_logits=self.w2_bi40(ts_pairs)
        
        return bi_logits.squeeze(-1)
    

    def initialize_model(self):
        print("init nfilter parameters.\n")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)


class KFilter_Multihead(nn.Module):
    def __init__(self, n_heads=5, head_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(262, n_heads * head_dim)
        self.k_proj = nn.Linear(262, n_heads * head_dim)
        self.v_proj = nn.Linear(262, n_heads * head_dim)
        self.w_o = nn.Linear(n_heads * head_dim, 262)
        
        self.fc1 = nn.Linear(524, 524 * 4)
        self.fc2 = nn.Linear(524 * 4, 524)

        self.w_predict1 = nn.Linear(524, 262)
        self.w_predict2 = nn.Linear(262, 1)


    def forward_bi(self, target_embeddings:torch.Tensor, source_embeddings:torch.Tensor, k:int):
        q = self.q_proj(source_embeddings).view(1, self.n_heads, self.head_dim)
        k = self.k_proj(target_embeddings).view(target_embeddings.shape[0], self.n_heads, self.head_dim)
        v = self.v_proj(target_embeddings).view(target_embeddings.shape[0], self.n_heads, self.head_dim)
        
        attn_scores = torch.einsum('mhd,nhd->nhm', q, k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended_values = torch.einsum('nhm,nhd->nhd', attn_weights, v)
        attended_values = attended_values.view(attended_values.shape[0], -1)
        output = self.w_o(attended_values)
        output = F.dropout(output,p=0.1,training=self.training)

        qt = self.q_proj(target_embeddings).view(target_embeddings.shape[0], self.n_heads, self.head_dim)
        kt = self.k_proj(source_embeddings).view(1, self.n_heads, self.head_dim)
        vt = self.v_proj(source_embeddings).view(1, self.n_heads, self.head_dim)
        
        attn_scorest = torch.einsum('nhd,mhd->nhm', qt, kt) / math.sqrt(self.head_dim)
        attn_weightst = F.softmax(attn_scorest, dim=-1)
        attended_valuest = torch.einsum('nhm,mhd->nhd', attn_weightst, vt)
        attended_valuest = attended_valuest.view(attended_valuest.shape[0], -1)
        outputt = self.w_o(attended_valuest)  # (N,262)
        outputt = F.dropout(outputt,p=0.1,training=self.training)

        output = torch.cat([output, outputt], dim=-1)
        
        r = output
        output = self.fc2(F.relu(self.fc1(output)))
        output = F.dropout(output,p=0.1,training=self.training) + r

        output = F.dropout(F.relu(self.w_predict1(output)),p=0.1,training=self.training)
        output = self.w_predict2(output)

        return output.squeeze(-1)
    
    def initialize_model(self):
        print("init kfilter_multihead.")
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)

        nn.init.xavier_uniform_(self.w_o.weight, gain=1.0)
        nn.init.zeros_(self.w_o.bias)

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

        nn.init.xavier_uniform_(self.w_predict1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.w_predict1.bias)
        nn.init.xavier_uniform_(self.w_predict2.weight, gain=1.0)
        nn.init.zeros_(self.w_predict2.bias)


class GraphModelContrastivev5(nn.Module):
    def __init__(self,
                 h_token=768,
                 gat_nhead=1,
                 node_transform_nhead=1,node_transform_num_layers=1,
                 h_st=5,
                 fusion_type:str="Attention"
                ) -> None:
        super().__init__()
        self.gat_layer1=GATv2Conv(in_channels=h_st,out_channels=h_st,
                                  heads=gat_nhead,add_self_loops=False,concat=False,dropout=0.1)
        self.gat_layer2=GATv2Conv(in_channels=h_st,out_channels=h_st,
                                  heads=gat_nhead,add_self_loops=False,concat=False,dropout=0.1)
        
        node_transform_layer=nn.TransformerEncoderLayer(d_model=h_st,nhead=node_transform_nhead,
                                                        dim_feedforward=2*h_st,
                                                        activation="gelu",batch_first=True)
        self.node_transform=nn.TransformerEncoder(encoder_layer=node_transform_layer,
                                                  num_layers=node_transform_num_layers)

        self.fusion_type=fusion_type
        if fusion_type=="Attention":
            self.avg_w1=nn.Linear(3,6)
            self.avg_w2=nn.Linear(6,3)
            self.max_w1=nn.Linear(3,6)
            self.max_w2=nn.Linear(6,3)
            self.w_sm=nn.Linear(h_token,512)
            self.w_co=nn.Linear(512*5,251)

            self.sm_norm=nn.LayerNorm(512)
            self.st_norm=nn.LayerNorm(5)
            self.co_norm=nn.LayerNorm(251)
        elif fusion_type=="Union":
            self.w_sm=nn.Linear(h_token,512)
            self.w_co=nn.Linear(512*5,251)

            self.sm_norm=nn.LayerNorm(512)
            self.st_norm=nn.LayerNorm(5)
            self.co_norm=nn.LayerNorm(251)
        elif fusion_type=="Intersection":
            self.w_sm=nn.Linear(h_token,h_token)
            self.w_st=nn.Linear(5,h_token)
            self.w_co=nn.Linear(512*5,h_token)

            self.sm_norm=nn.LayerNorm(h_token)
            self.st_norm=nn.LayerNorm(h_token)
            self.co_norm=nn.LayerNorm(h_token)
        else:
            raise ValueError("Invalid fusion type.")
        
        self.w1_fuse=nn.Linear(h_token,int(h_token/2))

        self.w_metric=nn.Linear(h_token,1)


    def forward_visualize(self,change_embeddings:torch.Tensor,node_embeddings:torch.Tensor,
                          edge_indexs:torch.Tensor,
                          node_predict_indexs:torch.Tensor,
                          node_predict_types:torch.Tensor,
                          node_predict_labels:torch.Tensor,
                          change_node_indexs:torch.Tensor,
                          st_embeddings:torch.Tensor,
                          infonce_temperature:float=0.1):
        st_embeddings=self.forward_gtn(st_embeddings,edge_indexs)
        fuse_embeddings, x=self.fea_fuse(node_embeddings,st_embeddings)

        target_embeddings=fuse_embeddings[node_predict_indexs]
        source_embeddings=fuse_embeddings[change_node_indexs]

        loss_infonce,similarities=self.loss_fn_infonce_x(target_embeddings,source_embeddings,node_predict_types,
                                                         node_predict_labels,infonce_temperature)
        
        return loss_infonce,similarities,target_embeddings.detach(),source_embeddings.detach()
        
    
    def forward(self,change_embeddings:torch.Tensor,node_embeddings:torch.Tensor,
                edge_indexs:torch.Tensor,
                node_predict_indexs:torch.Tensor,
                node_predict_types:torch.Tensor,
                node_predict_labels:torch.Tensor,
                change_node_indexs:torch.Tensor,
                st_embeddings:torch.Tensor,
                infonce_temperature:float=0.1):
        st_embeddings=self.forward_gtn(st_embeddings,edge_indexs)
        # fuse_embeddings, x=self.fea_fuse(node_embeddings,st_embeddings,ret_x="st")
        # fuse_embeddings, x=self.fea_fuse(node_embeddings,st_embeddings,ret_x="sm")
        # fuse_embeddings, x=self.fea_fuse(node_embeddings,st_embeddings,ret_x="co")
        fuse_embeddings, x=self.fea_fuse(node_embeddings,st_embeddings)

        # target_embeddings=x[node_predict_indexs]
        # source_embeddings=x[change_node_indexs]
        target_embeddings=fuse_embeddings[node_predict_indexs]
        source_embeddings=fuse_embeddings[change_node_indexs]

        # similarities=self.euc_similarity(target_embeddings,source_embeddings,node_predict_types)
        # similarities=self.cos_similarity(target_embeddings,source_embeddings,node_predict_types)

        loss_infonce,similarities=self.loss_fn_infonce_x(target_embeddings,source_embeddings,node_predict_types,
                                                         node_predict_labels,infonce_temperature)
        
        # return loss_infonce,euc_sim
        return loss_infonce,similarities,target_embeddings.detach(),source_embeddings.detach()
    

    def fea_fuse(self,x_sm:torch.Tensor,x_st:torch.Tensor,ret_x:str=None):
        N = x_sm.size(0)

        x_sm=self.w_sm(x_sm)
        x_co = torch.einsum('ni,nj->nij', x_sm, x_st)
        
        x_co = x_co.view(N, -1)
        x_co = self.w_co(x_co)

        if self.fusion_type=="Intersection":
            x_st=self.w_st(x_st)

        x_sm = self.sm_norm(x_sm)
        x_st = self.st_norm(x_st)
        x_co = self.co_norm(x_co)
        
        if self.fusion_type=="Attention":
            avg_sm = x_sm.mean(dim=1, keepdim=True)
            avg_st = x_st.mean(dim=1, keepdim=True)
            avg_co = x_co.mean(dim=1, keepdim=True)
            max_sm = x_sm.max(dim=1, keepdim=True).values
            max_st = x_st.max(dim=1, keepdim=True).values
            max_co = x_co.max(dim=1, keepdim=True).values
            
            avg_cat = torch.cat([avg_sm, avg_st, avg_co], dim=1)
            max_cat = torch.cat([max_sm, max_st, max_co], dim=1)
            
            avg_trans = self.avg_w2(F.leaky_relu(self.avg_w1(avg_cat)))
            max_trans = self.max_w2(F.leaky_relu(self.max_w1(max_cat)))
            
            channel_weights = avg_trans + max_trans
            
            x_sm_weighted = x_sm * channel_weights[:, 0].unsqueeze(1)
            x_st_weighted = x_st * channel_weights[:, 1].unsqueeze(1)
            x_co_weighted = x_co * channel_weights[:, 2].unsqueeze(1)
            x_fuse = torch.cat([x_sm_weighted, x_st_weighted, x_co_weighted], dim=1)
        elif self.fusion_type=="Union":
            x_fuse = torch.cat([x_sm, x_st, x_co], dim=1)
        elif self.fusion_type=="Intersection":
            x_fuse = torch.amax(torch.stack([x_sm, x_st, x_co]), dim=0)

        x_fuse = self.w1_fuse(x_fuse)
        x_fuse=F.dropout(F.leaky_relu(x_fuse),p=0.1,training=self.training)

        r=None
        if ret_x=="sm":
            r=x_sm
        elif ret_x=="st":
            r=x_st
        elif ret_x=="co":
            r=x_co

        return x_fuse, r


    def euc_similarity(self,target_embeddings,source_embeddings,node_predict_types):
        selected_source = source_embeddings[node_predict_types]
        
        diff = target_embeddings - selected_source
        
        euc_dis = 1.0/torch.sqrt(torch.sum(diff ** 2, dim=1))
        
        return euc_dis
    
    def cos_similarity(self,target_embeddings,source_embeddings,node_predict_types):
        selected_source = source_embeddings[node_predict_types]
        
        cos_sim = F.cosine_similarity(
            target_embeddings,
            selected_source,
            dim=1,
        )
        
        return cos_sim


    def loss_fn_infonce(self,target_embeddings:torch.Tensor,
                        source_embeddings:torch.Tensor,
                        node_predict_types:torch.Tensor,
                        node_predict_labels:torch.Tensor,
                        infonce_temperature:float):
        similarities = torch.zeros(target_embeddings.shape[0], 1, 
                                dtype=torch.float, device=target_embeddings.device)
        total_loss = 0.0
        num_valid_queries = 0
        
        for i, cur_type in enumerate(torch.unique(node_predict_types)):
            cur_indices = torch.where(node_predict_types == cur_type)[0]
            cur_source = source_embeddings[cur_type]
            cur_targets = target_embeddings[cur_indices]
            
            expanded_source = cur_source.unsqueeze(0).expand_as(cur_targets)
            concat_input = torch.cat([expanded_source, cur_targets], dim=1)
            cur_similarities = self.w_metric(concat_input)
            similarities[cur_indices] = cur_similarities
            
            cur_labels = node_predict_labels[cur_indices].squeeze(1)
            pos_mask = (cur_labels == 1)
            
            if not torch.any(pos_mask):
                continue
            
            logits = cur_similarities.squeeze(1) / infonce_temperature
            log_denominator = torch.logsumexp(logits, dim=0)
            log_numerator = torch.logsumexp(logits[pos_mask], dim=0)
            
            query_loss = log_denominator - log_numerator
            total_loss += query_loss
            num_valid_queries += 1
        
        if num_valid_queries > 0:
            total_loss = total_loss / num_valid_queries
        else:
            total_loss = torch.tensor(0.0, device=target_embeddings.device)

        return total_loss, similarities
    

    def loss_fn_infonce_x(self,target_embeddings:torch.Tensor,
                        source_embeddings:torch.Tensor,
                        node_predict_types:torch.Tensor,
                        node_predict_labels:torch.Tensor,
                        infonce_temperature:float):
        similarities = torch.zeros(target_embeddings.shape[0], 1, 
                                dtype=torch.float, device=target_embeddings.device)
        total_loss = 0.0
        num_valid_queries = 0
        
        for i, cur_type in enumerate(torch.unique(node_predict_types)):
            cur_indices = torch.where(node_predict_types == cur_type)[0]
            cur_source = source_embeddings[cur_type]
            cur_targets = target_embeddings[cur_indices]
            
            expanded_source = cur_source.unsqueeze(0).expand_as(cur_targets)
            cur_similarities = torch.sum(expanded_source * cur_targets, dim=1, keepdim=True)
            similarities[cur_indices] = cur_similarities
            
            cur_labels = node_predict_labels[cur_indices].squeeze(1)
            pos_mask = (cur_labels == 1)
            
            if not torch.any(pos_mask):
                continue
            
            logits = cur_similarities.squeeze(1) / infonce_temperature
            log_denominator = torch.logsumexp(logits, dim=0)
            log_numerator = torch.logsumexp(logits[pos_mask], dim=0)
            
            query_loss = log_denominator - log_numerator
            total_loss += query_loss
            num_valid_queries += 1
        
        if num_valid_queries > 0:
            total_loss = total_loss / num_valid_queries
        else:
            total_loss = torch.tensor(0.0, device=target_embeddings.device)

        return total_loss, similarities


    def forward_gtn(self,node_embeddings:torch.Tensor,
                    edge_indexs:torch.Tensor):
        node_embeddings=self.gat_layer1(node_embeddings,edge_indexs)
        node_embeddings=F.dropout(F.leaky_relu(node_embeddings),p=0.1,training=self.training)

        node_embeddings=self.gat_layer2(node_embeddings,edge_indexs)
        node_embeddings=F.dropout(F.leaky_relu(node_embeddings),p=0.1,training=self.training)

        # node_embeddings=self.node_transform(node_embeddings.unsqueeze(dim=0))
        return node_embeddings.squeeze()


    def initialize_model(self):
        print("init model parameters.\n")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                if module.bias is not None:
                    init.zeros_(module.bias)
            
            elif isinstance(module, GATv2Conv):
                init.kaiming_normal_(module.lin_l.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                init.kaiming_normal_(module.lin_r.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                if module.lin_l.bias is not None:
                    init.zeros_(module.lin_l.bias)
                if module.lin_r.bias is not None:
                    init.zeros_(module.lin_r.bias)
            
            elif isinstance(module, nn.LayerNorm):
                init.ones_(module.weight)
                init.zeros_(module.bias)

            elif isinstance(module, nn.TransformerEncoderLayer):
                init.xavier_uniform_(module.self_attn.in_proj_weight)
                init.xavier_uniform_(module.self_attn.out_proj.weight)
                if module.self_attn.in_proj_bias is not None:
                    init.zeros_(module.self_attn.in_proj_bias)
                if module.self_attn.out_proj.bias is not None:
                    init.zeros_(module.self_attn.out_proj.bias)
                init.kaiming_uniform_(module.linear1.weight, nonlinearity='leaky_relu', a=0.01)
                init.zeros_(module.linear1.bias)
                init.kaiming_uniform_(module.linear2.weight, nonlinearity='leaky_relu', a=0.01)
                init.zeros_(module.linear2.bias)
                init.ones_(module.norm1.weight)
                init.zeros_(module.norm1.bias)
                init.ones_(module.norm2.weight)
                init.zeros_(module.norm2.bias)
            
            elif isinstance(module, nn.TransformerEncoder):
                for sub_module in module.layers:
                    self.initialize_model_layer(sub_module)

    
    def initialize_model_layer(self, module):
        if isinstance(module, nn.TransformerEncoderLayer):
            init.xavier_uniform_(module.self_attn.in_proj_weight)
            init.xavier_uniform_(module.self_attn.out_proj.weight)
            if module.self_attn.in_proj_bias is not None:
                init.zeros_(module.self_attn.in_proj_bias)
            if module.self_attn.out_proj.bias is not None:
                init.zeros_(module.self_attn.out_proj.bias)
            init.kaiming_uniform_(module.linear1.weight, nonlinearity='leaky_relu', a=0.01)
            init.zeros_(module.linear1.bias)
            init.kaiming_uniform_(module.linear2.weight, nonlinearity='leaky_relu', a=0.01)
            init.zeros_(module.linear2.bias)
            init.ones_(module.norm1.weight)
            init.zeros_(module.norm1.bias)
            init.ones_(module.norm2.weight)
            init.zeros_(module.norm2.bias)