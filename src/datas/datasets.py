from torch.utils.data import Dataset
from omegaconf import DictConfig
from datas.graphs import IDG
from datas.samples import IDGSample
from os.path import exists
import json
from vocabulary import Vocabulary
from transformers import AutoTokenizer


class IDGDataset(Dataset):
    # def __init__(self, IDG_paths_json: str, config: DictConfig, vocab: Vocabulary, tokenizer: AutoTokenizer, max_length: int=512) -> None:
    def __init__(self, IDG_paths_json: str, config: DictConfig, tokenizer: AutoTokenizer, max_length: int=512) -> None:
        """
        Args:
            IDG_root_path: json file of list of IDG paths
        """
        super().__init__()
        self.__config = config
        assert exists(IDG_paths_json), f"{IDG_paths_json} not exists!"
        with open(IDG_paths_json, "r") as f:
            __IDG_paths_all = json.load(f)
        # self.__vocab = vocab
        self.__tokenizer = tokenizer
        self.__max_length = max_length
        self.__IDGs = list()
        indices = __IDG_paths_all["filename"].keys()
        IDGs_attr = []
        for index in indices:
            IDGs_attr.append([
                __IDG_paths_all["filename"][index],
                __IDG_paths_all["node_feature"][index],
                __IDG_paths_all["node_target"][index],
                __IDG_paths_all["edges"][index],
                __IDG_paths_all["node_lines"][index],
                #新添加target，以target为目标label
                # __IDG_paths_all["target"][index],
            ])
        for idg_attr in IDGs_attr:
            idg = IDG(attr=idg_attr, dataset_name=config.dataset.name)
            # if len(idg.nodes) != 0:
            self.__IDGs.append(idg)
        self.__n_samples = len(self.__IDGs)

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> IDGSample:
        idg: IDG = self.__IDGs[index]
        tokenized = self.__tokenizer(idg.code, padding="max_length", truncation=True, max_length=self.__max_length, return_tensors="pt")
        return IDGSample(graph=idg.data, label=idg.label, code_id=tokenized.input_ids.squeeze(0), attention_mask=tokenized.attention_mask.squeeze(0))

    def get_n_samples(self):
        return self.__n_samples
