from argparse import ArgumentParser
from typing import cast

from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from datas.datamodules import IDGDataModule
from model.vd import GEHVD
from train import train
from vocabulary import Vocabulary


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default=r"/home/wyx/VulExplain/src/configs/dwk.yaml",
                            type=str)
    return arg_parser


def vul_detect(config_path: str):
    config = cast(DictConfig, OmegaConf.load(config_path))
    print_config(config, ["dataset", "gnn", "classifier", "hyper_parameters"])
    seed_everything(config.seed, workers=True)

    # vocab = Vocabulary.build_from_w2v(config.gnn.w2v_path)
    # vocab_size = vocab.get_vocab_size()
    # pad_idx = vocab.get_pad_id()

    # Init datamodule
    # data_module = IDGDataModule(config, vocab)
    data_module = IDGDataModule(config)

    # Init model
    # model = GEHVD(config, vocab, vocab_size, pad_idx)
    model = GEHVD(config)

    train(model, data_module, config)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    vul_detect(__args.config)
