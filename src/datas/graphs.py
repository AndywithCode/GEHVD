from dataclasses import dataclass
from typing import List
import torch, os
from torch_geometric.data import Data
from vocabulary import Vocabulary


@dataclass(frozen=True)
class IDGNode:
    ln: int


@dataclass
class IDGEdge:
    from_node: IDGNode
    to_node: IDGNode


@dataclass
class IDG:
    def __init__(self, attr: tuple = None, dataset_name: str='Libav'):
        self.__attr = attr
        self.__dataset = dataset_name
        self.__data = Data(x=torch.tensor(attr[1], dtype=torch.float), edge_index=torch.tensor(attr[3], dtype=torch.long))
        self.__init_graph()
        self.__init_statement()
    
    def __init_statement(self):
        linesnumbers = self.__attr[4]
        linesnumbers = list(set(linesnumbers))
        if -1 in linesnumbers:
            linesnumbers.remove(-1)
        linesnumbers.sort()

        filename = self.__attr[0]
        with open(os.path.join(r'source_code/'+self.__dataset, filename), 'r', encoding="utf-8") as f:
            codes = f.readlines()
        self.__statement = str()
        for linenumber in linesnumbers:
            self.__statement += codes[linenumber-1]

    def __init_graph(self):
        self.__edges= list()
        for edges in self.__attr[3]:
            temp = list()
            for node in edges:
                temp.append(self.__attr[4][node])
            self.__edges.append(temp)
        if all(x == 0 for x in self.__attr[2]):
            self.__label = 0
        else:
            self.__label = 1
        # self.__label = self.__attr[5]

    @property
    def nodes(self) -> List[int]:
        return self.__attr[4]

    @property
    def edges(self) -> List[List[int]]:
        return self.__edges

    @property
    def label(self) -> int:
        return self.__label
    
    @property
    def data(self) -> Data:
        return self.__data
    
    @property
    def code(self) -> str:
        return self.__statement

    def to_torch(self, vocab: Vocabulary, max_len: int) -> Data:
        """Convert this graph into torch-geometric graph

        Args:
            vocab:
            max_len: vector max_len for node content
        Returns:
            :torch_geometric.data.Data
        """
        node_tokens = []
        for idx, n in enumerate(self.nodes):
            node_tokens.append(self.__tokens_list[idx])
        # [n_node, max seq len]
        node_ids = torch.full((len(node_tokens), max_len),
                              vocab.get_pad_id(),
                              dtype=torch.long)
        for tokens_idx, tokens in enumerate(node_tokens):
            ids = vocab.convert_tokens_to_ids(tokens)
            less_len = min(max_len, len(ids))
            node_ids[tokens_idx, :less_len] = torch.tensor(ids[:less_len],
                                                           dtype=torch.long)
        edge_index = torch.tensor(list(
            zip(*[[self.__node_to_idx[e.from_node],
                   self.__node_to_idx[e.to_node]] for e in self.edges])),
            dtype=torch.long)

        # edge_attr = torch.randn((edge_index.size(1), edge_feature_dim))

        # save token to `x` so Data can calculate properties like `num_nodes`
        return Data(x=node_ids, edge_index=edge_index)
        # return Data(x=node_ids, edge_index=edge_index, edge_attr=edge_attr, y=self.label)
