from torch import nn, Tensor

    
class StructureModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass

    def forward(self, single_representation: Tensor, pair_representation: Tensor):
        pass
