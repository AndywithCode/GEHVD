from dataclasses import dataclass
from typing import List
from torch_geometric.data import Data, Batch
import torch


@dataclass
class IDGSample:
    graph: Data
    label: int
    code_id: torch.Tensor
    attention_mask: torch.Tensor


class IDGBatch:
    def __init__(self, IDGs: List[IDGSample]):

        self.labels = torch.tensor([IDG.label for IDG in IDGs], dtype=torch.long)
        self.graphs = Batch.from_data_list([IDG.graph for IDG in IDGs])
        # self.graphs_edge = torch.stack([IDG.graph.edge_index for IDG in IDGs])
        self.code_ids = torch.stack([IDG.code_id for IDG in IDGs])
        self.atten_mask = torch.stack([IDG.attention_mask for IDG in IDGs])
        self.sz = len(IDGs)

    def __len__(self):
        return self.sz

    def pin_memory(self) -> "IDGBatch":
        self.labels = self.labels.pin_memory()
        self.graphs = self.graphs.pin_memory()
        self.code_ids = self.code_ids.pin_memory()
        self.atten_mask = self.atten_mask.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.graphs = self.graphs.to(device)
        self.code_ids = self.code_ids.to(device)
        self.atten_mask = self.atten_mask.to(device)