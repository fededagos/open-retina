from typing import Any, Iterable

import hydra.utils
import torch
from jaxtyping import Float, Int
from omegaconf import DictConfig

from openretina.data_io.base_dataloader import DataPoint
from openretina.models.core_readout import BaseCoreReadout
from openretina.utils.optimizer_utils import instantiate_optimizer, instantiate_scheduler


class TokenizedCoreReadout(BaseCoreReadout):
    """Core+readout model trained to predict external, frozen neural tokens.

    The readout emits d-dim conditioning [B, T_tok, N, d]; the shared head (on the readout)
    owns the loss and metrics. This model type replaces the scalar Poisson/correlation path.
    """

    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        hidden_channels: Iterable[int],
        n_neurons_dict: dict[str, int],
        core: DictConfig,
        readout: DictConfig,
        learning_rate: float = 0.001,
        data_info: dict[str, Any] | None = None,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
    ):
        core.channels = (in_shape[0], *hidden_channels)
        core_module = hydra.utils.instantiate(core, n_neurons_dict=n_neurons_dict)

        if "in_shape" not in readout:
            in_shape_readout = self.compute_readout_input_shape(in_shape, core_module)
            readout["in_shape"] = tuple(in_shape_readout)
        readout_module = hydra.utils.instantiate(readout, n_neurons_dict=n_neurons_dict)

        # Hydra configs for optimizer/scheduler; None -> tokenized defaults (see configure_optimizers).
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler

        self.save_hyperparameters()
        super().__init__(core=core_module, readout=readout_module, learning_rate=learning_rate, data_info=data_info)

    def forward(
        self, x: Float[torch.Tensor, "batch channels t h w"], data_key: str | None = None
    ) -> Float[torch.Tensor, "batch t_tok n d"]:
        return self.readout(self.core(x), data_key=data_key)

    @staticmethod
    def _assert_alignment(cond: torch.Tensor, targets: torch.Tensor) -> None:
        if cond.size(1) != targets.size(1):
            raise ValueError(
                f"Token-count mismatch: model produced T_tok={cond.size(1)} but targets have "
                f"{targets.size(1)}. Check frames_per_token / temporal aggregator config."
            )

    def _step(self, batch: tuple[str, DataPoint], stage: str) -> torch.Tensor:
        session_id, data_point = batch
        cond = self.forward(data_point.inputs, session_id)
        self._assert_alignment(cond, data_point.targets)

        head = self.readout.head
        loss = head.compute_loss(cond, data_point.targets)  # type: ignore
        reg = self.core.regularizer() + self.readout.regularizer(session_id)  # type: ignore
        total_loss = loss + reg

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        self.log(f"{stage}_total_loss", total_loss, on_step=False, on_epoch=True)
        for name, value in head.metrics(cond, data_point.targets).items():  # type: ignore
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss if stage == "train" else loss

    def training_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[str, DataPoint], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: tuple[str, DataPoint], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler, reusing openretina's Hydra-configurable
        optimization utilities (``instantiate_optimizer`` / ``instantiate_scheduler``).

        Both ``optimizer`` and ``lr_scheduler`` (passed to ``__init__``) default to ``None``.
        When they are ``None`` the tokenized defaults apply: AdamW plus a ``ReduceLROnPlateau``
        that *maximises* ``val_token_accuracy``. The generic openretina default scheduler
        monitors ``val_correlation``, which token models never log, so its default path is not
        reused here -- only the configured path is delegated to ``instantiate_scheduler``.
        """
        optimizer = instantiate_optimizer(self.optimizer_config, self.parameters(), self.learning_rate)

        if self.lr_scheduler_config is not None:
            # self.trainer is a property that *raises* (not returns None) when unattached;
            # OneCycleLR uses it to auto-fill total_steps, others ignore it.
            try:
                trainer = self.trainer
            except RuntimeError:
                trainer = None
            scheduler_dict = instantiate_scheduler(
                self.lr_scheduler_config,
                optimizer,
                self.learning_rate,
                trainer=trainer,
            )
        else:
            lr_decay_factor = 0.3
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",  # monitor token accuracy (higher is better)
                factor=lr_decay_factor,
                patience=5,
                min_lr=self.learning_rate * (lr_decay_factor**3),
            )
            scheduler_dict = {"scheduler": scheduler, "monitor": "val_token_accuracy", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
