import copy
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.utils
import torch.utils.data

from MethodInfo import MethodInfo
from MutantInfo import MutantInfo
from util import extract_edges,expand_nodes


class ChangeImpactDataBuilder():
    def __init__(self,method_info:MethodInfo,mutant_info:MutantInfo):
        super().__init__()
        self.adj:torch.Tensor=method_info.adj
        self.embeddings:torch.Tensor=method_info.embeddings
        self.st_embeddings:torch.Tensor=method_info.st_embeddings
        self.n_node=self.adj.shape[0]
        self.n_edge=torch.sum(torch.sum(self.adj,dim=1),dim=0)

        self.change_embeddings:torch.Tensor=mutant_info.change_embeddings
        self.node_change_indexs:torch.Tensor=torch.tensor(mutant_info.change_method_indexs,dtype=torch.int)
        # self.n_change=self.change_embeddings.shape[0]
        self.n_change=mutant_info.n_change

        self.n_positive=0
        self.n_negitive=0

        self.node_indexs=[]
        self.node_types=[]
        self.node_labels=[]

        for change_index in range(self.n_change):
            positives=len(mutant_info.impacted_method_indexs[change_index])
            negitives=len(mutant_info.unimpacted_method_indexs[change_index])

            self.node_indexs.extend(mutant_info.impacted_method_indexs[change_index])
            self.node_labels.extend([1]*positives)
            self.node_types.extend([change_index]*(positives))

            self.node_indexs.extend(mutant_info.unimpacted_method_indexs[change_index])
            self.node_labels.extend([0]*negitives)
            self.node_types.extend([change_index]*(negitives))

            self.n_positive+=positives
            self.n_negitive+=negitives
            
        self.node_indexs=torch.tensor(self.node_indexs,dtype=torch.int)
        self.node_types=torch.tensor(self.node_types,dtype=torch.int)
        self.node_labels=torch.tensor(self.node_labels,dtype=torch.int)

        self.n_predict=self.n_change

    
    def build_batch_data(self, batch_mutant_indexs:torch.Tensor, batch_node_indexs:torch.Tensor):
        bool_filter=(
            torch.isin(self.node_indexs,batch_node_indexs)
            & torch.isin(self.node_types,batch_mutant_indexs)
        )
        node_predict_indexs=self.node_indexs[bool_filter]
        node_predict_labels=self.node_labels[bool_filter].reshape((-1,1))
        node_predict_types=self.node_types[bool_filter]
        mutant_indexs=node_predict_types[:].tolist()

        change_types=torch.unique(node_predict_types).sort()[0]
        change_embeddings=self.change_embeddings[change_types]
        node_change_indexs=self.node_change_indexs[change_types]
        type_mapping=torch.full((torch.max(change_types).item()+1,),-1)
        type_mapping[change_types]=torch.arange(change_types.shape[0])
        node_predict_types=type_mapping[node_predict_types]

        node_indexs=node_predict_indexs.tolist()
        node_indexs.extend(node_change_indexs.tolist())
        node_indexs=expand_nodes(self.adj,node_indexs,expand_size=2)
        node_indexs=list(set(node_indexs))
        node_indexs.sort()
        edge_indexs=extract_edges(self.adj,node_indexs,node_indexs)
        node_embeddings=self.embeddings[node_indexs]
        st_embeddings=self.st_embeddings[node_indexs]
        node_mapping=torch.full((max(node_indexs)+1,),-1)
        node_mapping[node_indexs]=torch.arange(len(node_indexs))
        node_predict_indexs_origin=node_predict_indexs[:]
        node_predict_indexs=node_mapping[node_predict_indexs]
        node_change_indexs=node_mapping[node_change_indexs]
        edge_indexs=node_mapping[edge_indexs]

        return change_embeddings,node_embeddings,edge_indexs,node_predict_indexs,node_predict_labels,node_predict_types,node_change_indexs,mutant_indexs,node_predict_indexs_origin,st_embeddings
        

    def info(self):
        print(f"{self.n_node} nodes")
        print(f"{self.n_change} mutants")
        print(f"{self.n_edge} edges")
        print(f"{self.n_positive} positives")
        print(f"{self.n_negitive} negitives\n")


    def get_negitive_positive_weight(self):
        return self.n_negitive/self.n_positive
    

    def __len__(self) -> int:
        return self.n_predict