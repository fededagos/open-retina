# openretina/modules/readout/tokenized/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int


class TokenHead(nn.Module):
    """Turns per-token conditioning [B, T_tok, N, d] into token predictions and owns the loss.

    Keeping the readout head-agnostic (it only emits d-dim conditioning) lets discrete
    (classifier/CE) and continuous (flow-matching) tokenizers share one readout code path.
    """

    def compute_loss(self, cond: Float[torch.Tensor, "B T_tok N d"], target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, cond: Float[torch.Tensor, "B T_tok N d"]) -> torch.Tensor:
        raise NotImplementedError

    def metrics(self, cond: Float[torch.Tensor, "B T_tok N d"], target: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}


class ClassifierTokenHead(TokenHead):
    """Discrete-token head: linear/MLP classifier over the codebook + cross-entropy."""

    def __init__(self, cond_dim: int, codebook_size: int, hidden: int | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.label_smoothing = label_smoothing
        if hidden is None:
            self.net: nn.Module = nn.Linear(cond_dim, codebook_size)
        else:
            self.net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.GELU(), nn.Linear(hidden, codebook_size))

    def logits(self, cond: Float[torch.Tensor, "B T_tok N d"]) -> Float[torch.Tensor, "B T_tok N K"]:
        return self.net(cond)

    def compute_loss(
        self, cond: Float[torch.Tensor, "B T_tok N d"], target: Int[torch.Tensor, "B T_tok N"]
    ) -> torch.Tensor:
        if target.is_floating_point():
            raise TypeError(
                "ClassifierTokenHead expects integer token codes as targets, got a floating-point tensor. "
                "(Codebook-index range is enforced by cross_entropy itself to avoid a per-step device sync.)"
            )
        logits = self.logits(cond)
        k = self.codebook_size
        return F.cross_entropy(logits.reshape(-1, k), target.reshape(-1).long(), label_smoothing=self.label_smoothing)

    def predict(self, cond: Float[torch.Tensor, "B T_tok N d"]) -> Int[torch.Tensor, "B T_tok N"]:
        return self.logits(cond).argmax(dim=-1)

    @torch.no_grad()
    def metrics(
        self, cond: Float[torch.Tensor, "B T_tok N d"], target: Int[torch.Tensor, "B T_tok N"]
    ) -> dict[str, torch.Tensor]:
        pred = self.predict(cond)
        return {"token_accuracy": (pred == target.long()).float().mean()}
