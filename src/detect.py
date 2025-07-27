# from argparse import ArgumentParser
import torch
from data_generator import build_PDG, build_IDG
# from model.VulExplain import CodeT5PlusWithGAT
from model.vd import GEHVD
from datas.graphs import IDG
from vocabulary import Vocabulary
import networkx as nx
from dataProcess.symbolizer import clean_gadget, tokenize_code_line
from torch_geometric.data import Batch
from typing import Tuple

def add_syms(idg: nx.DiGraph, split_token: bool, line_to_statement: dict) -> Tuple[nx.DiGraph, str]:
    """

    Args:
        idg:
        split_token:
        line_to_statement:

    Returns:

    """
    code_lines, ln_stat = list(), list()
    for n in idg.nodes:
        code_lines.append(line_to_statement[n])
        ln_stat.append((n, line_to_statement[n]))

    sym_code_lines = clean_gadget(code_lines)
    for idx, n in enumerate(idg):
        idg.nodes[n]["code_sym_token"] = tokenize_code_line(sym_code_lines[idx], split_token)

    stats_list = [s for _, s in sorted(ln_stat, key=lambda x: x[0])]
    stats = "".join(stats_list)
    return idg, stats

if __name__ == '__main__':
    # __arg_parser = ArgumentParser()
    # __arg_parser.add_argument("-c",
    #                           "--check-point",
    #                           help="checkpoint path",
    #                           type=str)
    # __arg_parser.add_argument("-t",
    #                           "--target",
    #                           help="code csv root path",
    #                           type=str)
    # __arg_parser.add_argument("-s",
    #                           "--source",
    #                           help="source code path",
    #                           type=str)
    # __args = __arg_parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # load model
    model = GEHVD.load_from_checkpoint(checkpoint_path='').to(device)
    # # load config and vocab
    # config = model.hparams["config"]
    # vocab = model.hparams["vocab"]
    # vocab_size = vocab.get_vocab_size()
    # pad_idx = vocab.get_pad_id()
    # preprocess for the code to detect
    vocab = Vocabulary.build_from_w2v('w2v.wv')
    PDG, key_line_map = build_PDG(r"sensiAPI.txt",
                                  r'1-pdg')
    idg_dict, line_to_statement = build_IDG(r'7785.c', PDG, key_line_map)
    Datas, meta_datas = list(), list()
    idx_to_idg = dict()
    ct = 0
    for k in idg_dict:
        for idg in idg_dict[k]:
            idg_sym, idg_stats = add_syms(idg, False, line_to_statement)
            Datas.append((IDG(idg=idg_sym).to_torch(vocab, 16), idg_stats))
            meta_datas.append(idg_sym.graph["key_line"])
            idx_to_idg[ct] = idg_sym
            ct += 1
    # predict
    batch = Batch.from_data_list(Datas).to(device)
    # logits = model(batch)
    # _, preds = logits.max(dim=1)
    # batched_res = zip(meta_datas, preds.tolist())
    # for res in batched_res:
    #     if res[1] == 1:
    #         print(res[0])
