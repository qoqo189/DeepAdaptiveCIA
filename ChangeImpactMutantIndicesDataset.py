import torch
from torch.utils.data import Dataset


class ChangeImpactMutantIndicesDataset(Dataset):
    def __init__(self,mutant_indexs:list):
        super().__init__()
        self.mutant_indexs=torch.tensor(mutant_indexs)
        self.n_change=len(mutant_indexs)

    
    def info(self):
        print(f"{self.n_change} changes\n")


    def __getitems__(self, indices: list):
        mutant_indexs=self.mutant_indexs[indices]
        return mutant_indexs


    def __len__(self) -> int:
        return self.n_change