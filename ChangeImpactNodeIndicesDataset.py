import torch
from torch.utils.data import Dataset

from MethodInfo import MethodInfo

from util import method_name_match


class ChangeImpactNodeIndicesDataset(Dataset):
    def __init__(self,method_info:MethodInfo):
        super().__init__()

        self.node_indexs=[]
        for method in method_info.exec_record_dt["case"]:
            if method not in method_info.exec_record_index_mapping:
                raise FileNotFoundError(f"method {method} not found.\n")
            self.node_indexs.append(method_info.exec_record_index_mapping[method])
        self.node_indexs=torch.tensor(self.node_indexs,dtype=torch.int)
        self.n_predict_node=len(self.node_indexs)

    
    def info(self):
        print(f"{self.n_predict_node} test nodes\n")


    def __getitems__(self, indices: list):
        node_indexs=self.node_indexs[indices]
        return node_indexs


    def __len__(self) -> int:
        return self.n_predict_node