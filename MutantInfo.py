import random
import math
import time
import os

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

import torch

from MethodInfo import MethodInfo
from config import *
from util import method_name_match, search_paths

class MutantInfo:
    def __init__(self, method_info:MethodInfo,
                 mutant_info_dir:str=mutant_info_dir,
                 mutant_execrecord_home:str=mutant_execrecord_home,
                 load_embeddings:bool=True,
                 debug:bool=False) -> None:
        self.not_exist=set([
            "org.apache.commons.collections4.trie.PatriciaTrieTest.initializationError()",
            "org.apache.commons.collections4.trie.UnmodifiableTrieTest.initializationError()",
            "junit.framework.TestSuite$1.warning()",
            "org.junit.runner.manipulation.Filter.initializationError()"
        ])
        self.not_found_methods=set()
        self.method_info=method_info
        
        self.ids, self.change_methods=self.get_mutant_info(mutant_info_dir)

        if debug:
            self.n_change=1000
        else:
            self.n_change=len(self.ids)

        if load_embeddings:
            self.change_embeddings:torch.Tensor=torch.load(change_embeddings_dir)

        self.change_method_indexs=[]
        self.impacted_method_indexs=[]
        self.unimpacted_method_indexs=[]

        self.method_indexs_on_path=dict()
        self.method_change_history=dict()

        for change_index in range(self.n_change):
            mutant_id=self.ids[change_index]
            change_method=self.change_methods[change_index]

            change_method_index=method_name_match(change_method,method_info.nodes,method_info.simple_names)
            if change_method_index==-1:
                raise FileNotFoundError(f"method {change_method} not found.\n")
            self.change_method_indexs.append(change_method_index)
            
            mutant_exec_record_dir=mutant_execrecord_home+f"{path_sep}"+mutant_id+".xml"

            impacted_methods,impacted_method_indexs,unimpacted_methods,unimpacted_method_indexs=self.get_methods(mutant_exec_record_dir,method_info)
            self.impacted_method_indexs.append(impacted_method_indexs)
            self.unimpacted_method_indexs.append(unimpacted_method_indexs)

            if change_method_index not in self.method_indexs_on_path:
                impacted_node_indexs, impacted_node_path, unimpacted_node_indexs, unimpacted_node_path=search_paths(
                    method_info.adj,change_method_index,
                    impacted_method_indexs,unimpacted_method_indexs,
                    callpath_search_max_depth,
                    method_info.n_node
                )
                impacted_filtered = [idx for idx, path in zip(impacted_node_indexs, impacted_node_path) if path <= callpath_search_max_depth]
                unimpacted_filtered = [idx for idx, path in zip(unimpacted_node_indexs, unimpacted_node_path) if path <= callpath_search_max_depth]
                all_filtered = set(sorted(impacted_filtered + unimpacted_filtered, key=lambda x: x))
                self.method_indexs_on_path[change_method_index]=all_filtered
        
        self.n_positive=0
        self.n_negitive=0
        for i in range(self.n_change):
            self.n_positive+=len(self.impacted_method_indexs[i])
            self.n_negitive+=len(self.unimpacted_method_indexs[i])

    
    def parse_mutant_impact_history(self,mutant_indexs:list):
        for mutant_index in mutant_indexs:
            mutant_id=self.ids[mutant_index]
            change_method=self.change_methods[mutant_index]

            change_method_index=method_name_match(change_method,self.method_info.nodes,self.method_info.simple_names)
            
            mutant_exec_record_dir=mutant_execrecord_home+f"{path_sep}"+mutant_id+".xml"

            impacted_methods,impacted_method_indexs,unimpacted_methods,unimpacted_method_indexs=self.get_methods(mutant_exec_record_dir,self.method_info)
            if change_method_index not in self.method_change_history:
                self.method_change_history[change_method_index]=set()
            self.method_change_history[change_method_index].update(set(impacted_method_indexs))

    
    def get_candidate_node(self,mutant_index:int,
                           in_callgraph:bool,
                           in_history:bool):
        candidate_indexs=set()
        if in_callgraph:
            candidate_indexs.update(self.method_indexs_on_path[self.change_method_indexs[mutant_index]])
        if in_history:
            if self.change_method_indexs[mutant_index] in self.method_change_history:
                candidate_indexs.update(self.method_change_history[self.change_method_indexs[mutant_index]])
        return list(candidate_indexs)


    def get_methods(self,file_dir:str,method_info:MethodInfo):
        try:
            record_root=ET.parse(file_dir).getroot()
        except Exception as e:
            print(file_dir)
        exec_method_mapping={node.get("id"):node.get("value") for node in record_root.findall(".//compression-entries/entry")}

        impacted_methods=[]
        for failing_node in record_root.findall(".//failing/case"):
            failing_index=failing_node.text
            failing=exec_method_mapping[failing_index]
            if (failing not in method_info.exec_record_dt["failing"]) and (failing in method_info.exec_record_dt["case"]):
                impacted_methods.append(failing)

        impacted_methods=list(set(impacted_methods))

        unimpacted_methods=[case for case in method_info.exec_record_dt["case"] if case not in impacted_methods]
        unimpacted_methods=list(set(unimpacted_methods))

        impacted_method_indexs=[method_info.exec_record_index_mapping[method] for method in impacted_methods]
        unimpacted_method_indexs=[method_info.exec_record_index_mapping[method] for method in unimpacted_methods]
        impacted_method_indexs=list(set(impacted_method_indexs))
        unimpacted_method_indexs=list(set(unimpacted_method_indexs).difference(impacted_method_indexs))

        return impacted_methods,impacted_method_indexs,unimpacted_methods,unimpacted_method_indexs
    

    def get_mutant_info(self,file_dir:str)->list:
        mutant_root=ET.parse(file_dir)
        xml_mutants=[mutant for mutant in mutant_root.findall(".//mutant") if mutant.get("viable")=="true"]
        ids=[]
        change_methods=[]
        record_mutants=np.load(mutant_record_dir).tolist()
        for xml_mutant in xml_mutants:
            if not (xml_mutant.get("id") in record_mutants):
                continue
            ids.append(xml_mutant.get("id"))
            change_methods.append(xml_mutant.get("in"))
        return ids,change_methods
    

    def info(self):
        print(f"{self.n_change} changes")
        print(f"{self.n_positive} positives")
        print(f"{self.n_negitive} negitives\n")