import pandas as pd
import numpy as np
import torch
from config import *
from info_nce import InfoNCE
import re
import random

    
def expand_nodes(adj:torch.Tensor,nodes:list,expand_size:int):
    expanded_nodes=set(nodes)
    for _ in range(expand_size):
        neighbors=set(
            torch.nonzero(adj[list(expanded_nodes)])[:,1].tolist()
        )
        expanded_nodes.update(neighbors)
    return list(expanded_nodes)
    

def extract_edges(adj:torch.Tensor,from_nodes:list,to_nodes:list):
    relative_edges=torch.nonzero(adj[from_nodes,:][:,to_nodes])
    edge_index = torch.stack([
        torch.tensor(from_nodes,dtype=torch.int)[relative_edges[:, 0]],
        torch.tensor(to_nodes,dtype=torch.int)[relative_edges[:, 1]]
    ])
    return edge_index


def my_collate_fn(batch):
    return batch


def infoNCELoss(embeddings:torch.Tensor,labels:torch.Tensor):
    loss=InfoNCE()
    positive_keys=embeddings[labels==1]
    negitive_keys=embeddings[labels==0]

    if len(positive_keys) == 0 or len(negitive_keys) == 0:
        return torch.tensor(0.0, device=embeddings.device)

    permuted_indices = torch.randperm(len(positive_keys))
    query = positive_keys[permuted_indices]
    
    cur_loss=loss(query,positive_keys,negitive_keys)
    return cur_loss


def extract_method_parameters(method_signature):
    match = re.search(r'\((.*)\)', method_signature)
    if match:
        parameters = match.group(1)
        parameter_list = [param.strip() for param in parameters.split(',')]
        return parameter_list
    else:
        return []
    

def replace_generics_with_object(method_name):
    gen_types = ["T", "O", "K", "V", "E","L","M","R"]
    gen_types_arr=[]
    for gen_type in gen_types:
        gen_types_arr.append(gen_type+"[]")

    params=extract_method_parameters(method_name)
    params_replace=[]
    for param in params:
        if param in gen_types:
            params_replace.append("java.lang.Object")
        elif param in gen_types_arr:
            params_replace.append("java.lang.Object[]")
        else:
            params_replace.append(param)
    method_name=method_name[:method_name.find("(")]+f"({','.join(params_replace)})"
    return method_name


def extract_simple_name(full_name:str):
    method_name=full_name[:full_name.find("(")]
    simple_name=method_name[method_name.rfind(".")+1:]
    params=full_name[full_name.find("("):]
    simple_name=simple_name+params
    return simple_name


def method_name_match(name_xml:str,names_csv:list,simple_names_csv:list):
    name_xml_tran=replace_generics_with_object(name_xml)
    if name_xml in names_csv:
        return names_csv.index(name_xml)
    if name_xml_tran in names_csv:
        return names_csv.index(name_xml_tran)
    simple_name_xml=extract_simple_name(name_xml)
    simple_name_xml_tran=extract_method_parameters(name_xml_tran)
    if simple_name_xml in simple_names_csv:
        simple_name_xml_idx=[i for i,v in enumerate(simple_names_csv) if v==simple_name_xml]
        if len(simple_name_xml_idx)>1:
            return -1
        else:
            return simple_names_csv.index(simple_name_xml)
    if simple_name_xml_tran in simple_names_csv:
        simple_name_xml_tran_idx=[i for i,v in enumerate(simple_names_csv) if v==simple_name_xml_tran]
        if len(simple_name_xml_tran_idx)>1:
            return -1
        else:
            return simple_names_csv.index(simple_name_xml_tran)
    return -1


def search_paths(adj:torch.Tensor,source:int,
                 target_impacted:list,target_unimpacted:list,
                 max_depth:int,
                 n_node:int):
    impacted_node_paths={method:max_depth+1 for method in target_impacted}
    unimpacted_node_paths={method:max_depth+1 for method in target_unimpacted}
    dfs(adj,source,
        target_impacted,target_unimpacted,
        impacted_node_paths,unimpacted_node_paths,
        max_depth)
    impacted_node_indexs, impacted_node_path_coes=extract_node_paths(impacted_node_paths,n_node)
    unimpacted_node_indexs, unimpacted_node_path_coes=extract_node_paths(unimpacted_node_paths,n_node)
    return impacted_node_indexs, impacted_node_path_coes, unimpacted_node_indexs, unimpacted_node_path_coes


def extract_node_paths(node_paths:dict,n_node:int):
    if len(node_paths)==0:
        return [],[]
    node_indexs, path_lens=zip(*node_paths.items())
    node_indexs=list(node_indexs)
    path_lens=list(path_lens)
    return node_indexs, path_lens


def dfs(adj:torch.Tensor,source:int,
        target_impacted:list,target_unimpacted:list,
        impacted_node_paths:dict,unimpacted_node_paths:dict,
        max_depth:int,cur_path:set=set())->None:
    if max_depth==0:
        return
    cur_path.add(source)
    max_depth-=1
    if source in target_impacted:
        impacted_node_paths[source]=min(impacted_node_paths[source],len(cur_path))
    if source in target_unimpacted:
        unimpacted_node_paths[source]=min(unimpacted_node_paths[source],len(cur_path))
    neighbors=torch.nonzero(adj[:,source]).reshape(-1)
    for neighbor in neighbors:
        if neighbor.item() not in cur_path:
            dfs(adj,neighbor.item(),
                target_impacted,target_unimpacted,
                impacted_node_paths,unimpacted_node_paths,
                max_depth,cur_path)
    cur_path.remove(source)