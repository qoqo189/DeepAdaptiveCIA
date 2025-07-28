import pandas as pd
import torch
import numpy as np
import xml.etree.ElementTree as ET

from config import *
from util import method_name_match

class MethodInfo:
    def __init__(self,
                 method_dir:str=method_dir,
                 exec_record_dir:str=exec_record_dir,
                 not_found_mapping_dir:str=not_found_mapping_dir,
                 callgraph_pt_dir:str=callgraph_pt_dir,
                 load_embeddings:bool=True) -> None:
        self.not_exist=set([
            "org.apache.commons.collections4.trie.PatriciaTrieTest.initializationError()",
            "org.apache.commons.collections4.trie.UnmodifiableTrieTest.initializationError()",
            "junit.framework.TestSuite$1.warning()",
            "org.junit.runner.manipulation.Filter.initializationError()"
        ])
        self.load_embeddings=load_embeddings

        if self.load_embeddings:
            self.embeddings:torch.Tensor=torch.load(method_embedding_dir)
            self.st_embeddings:torch.Tensor=torch.load(st_embedding_pt_dir)
        self.adj:torch.Tensor=torch.load(callgraph_pt_dir)
        self.n_node=self.adj.shape[0]

        self.nodes,self.simple_names,self.not_found_mapping=self.get_methods(method_dir,not_found_mapping_dir)
        
        self.exec_record_dt=self.get_exec_record(exec_record_dir)
        self.exec_record_index_mapping=self.get_exec_record_mapping()
    
    def get_methods(self,file_dir:str,not_found_mapping_dir:str):
        method_df=pd.read_csv(file_dir,encoding="utf-8")
        nodes=method_df["method name"].tolist()
        simple_names=method_df["simple name"].tolist()
        not_found_df=pd.read_csv(not_found_mapping_dir)
        source_list=not_found_df["source"].tolist()
        target_list=not_found_df["target"].tolist()
        not_found_mapping=dict()
        for i in range(len(source_list)):
            not_found_mapping[source_list[i]]=target_list[i]
        return nodes,simple_names,not_found_mapping
    

    def get_exec_record(self,file_dir:str)->dict:
        record_root=ET.parse(file_dir).getroot()
        exec_record_dt={
            "case":[case.get("name") for case in record_root.findall(".//callings/test") if case.get("name") not in self.not_exist],  # 所有测试方法
            "failing":[failing.text for failing in record_root.findall(".//failing/case") if failing.get("name") not in self.not_exist],
        }
        return exec_record_dt
    

    def get_exec_record_mapping(self):
        exec_record_index_mapping=dict()
        for method in self.exec_record_dt["case"]:
            if method in self.not_exist:
                continue
            idx=method_name_match(method,self.nodes,self.simple_names)
            if idx==-1:
                if method not in self.not_found_mapping:
                    raise FileNotFoundError(f"method {method} not found.\n")
                idx=method_name_match(self.not_found_mapping[method],self.nodes,self.simple_names)
                if idx==-1:
                    raise FileNotFoundError(f"method {self.not_found_mapping[method]} not found.\n")
            exec_record_index_mapping[method]=idx
        return exec_record_index_mapping
    

    def info(self):
        print(f"n_node: {self.n_node}")
        if self.load_embeddings:
            print(f"method_embeddings shape: {self.embeddings.shape}")
        print(f"adj shape: {self.adj.shape}\n")