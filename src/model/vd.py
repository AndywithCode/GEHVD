from torch import nn
from omegaconf import DictConfig
import torch
from datas.samples import IDGBatch
from typing import Dict
from pytorch_lightning import LightningModule
from model.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder, GraphAttentionEncoder
from torch.optim import Adam, SGD, Adamax, RMSprop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch.nn.functional as F
from model.metrics import Statistic
from torch_geometric.data import Batch
from vocabulary import Vocabulary
from transformers import AutoModelForSeq2SeqLM


class GEHVD(LightningModule):
    r"""vulnerability detection model to detect vulnerability

    Args:
        config (DictConfig): configuration for the model
        vocabulary_size (int): the size of vacabulary
        pad_idx (int): the index of padding token
    """

    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    _gnns = {
        "gcn": GraphConvEncoder,
        "ggnn": GatedGraphConvEncoder,
        "gat": GraphAttentionEncoder
    }

    # _encoders = {
    #     "plus": 
    # }

    # def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int, pad_idx: int):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.__config = config
        hidden_size = 512
        # self.__graph_encoder = self._gnns["gcn"](config.gnn, vocab, vocabulary_size, pad_idx)
        self.__graph_encoder = self._gnns["gcn"](config.gnn)
        self.__encoder = AutoModelForSeq2SeqLM.from_pretrained(config.encoder.path, trust_remote_code=True).encoder
        # hidden layers
        layers = [
            nn.Linear(config.gnn.hidden_size + 1024, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier.drop_out)
        ]
        if config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.classifier.n_hidden_layers})")
        for _ in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.__classifier = nn.Linear(hidden_size, config.classifier.n_classes)

    def forward(self, batch_graph: Batch, batch_code_ids: torch.Tensor, batch_attention_mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            batch (Batch): [n_IDG (Data)]

        Returns: classifier results: [n_method; n_classes]
        """
        # [n_IDG, hidden size]
        graph_hid = self.__graph_encoder(batch_graph)
        encoder_outputs = self.__encoder(input_ids=batch_code_ids, attention_mask=batch_attention_mask)
        encoder_hid = encoder_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token's representation
        # Combine outputs
        # [1024, 4] cat [256, 4] = [1280, 4]
        combined_hid = torch.cat([encoder_hid, graph_hid], dim=1)
        hiddens = self.__hidden_layers(combined_hid)
        return self.__classifier(hiddens)

    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    def configure_optimizers(self) -> Dict:
        optimizer = self._get_optimizer(self.__config.hyper_parameters.optimizer)(
            self.parameters(), self.__config.hyper_parameters.learning_rate)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma ** epoch)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', 
            factor=0.1, 
            patience=3, 
            verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def _log_training_step(self, results: Dict):
        self.log_dict(results, on_step=True, on_epoch=False)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        logits = self(batch.graphs, batch.code_ids, batch.atten_mask)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"train_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="train")
            result.update(batch_metric)
            self._log_training_step(result)
            self.log("F1",
                     batch_metric["train_f1"],
                     prog_bar=True,
                     logger=False)
        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        logits = self(batch.graphs, batch.code_ids, batch.atten_mask)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"val_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="val")
            result.update(batch_metric)
        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        logits = self(batch.graphs, batch.code_ids, batch.atten_mask)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"test_loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="test")
            result.update(batch_metric)

        return {"loss": loss, "statistic": statistic}

    # ========== EPOCH END ==========
    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, group: str):
        log = self._prepare_epoch_end_log(step_outputs, group)
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in step_outputs])
        log.update(statistic.calculate_metrics(group))
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
