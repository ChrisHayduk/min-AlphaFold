"""
Full definition of the AlphaFold2 model, all of it in this single file.
"""
from torch import nn, Tensor
class Evoformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, msa_representation: Tensor, pair_representation: Tensor):
        pass
    
class StructureModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, single_representation: Tensor, pair_representation: Tensor):
        pass

class AlphaFold2(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, msa_representation: Tensor, pair_representation: Tensor):
        pass