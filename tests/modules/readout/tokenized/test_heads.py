# tests/modules/readout/tokenized/test_heads.py
import pytest
import torch

from openretina.modules.readout.tokenized.heads import ClassifierTokenHead


def test_compute_loss_is_finite_scalar():
    head = ClassifierTokenHead(cond_dim=6, codebook_size=16)
    cond = torch.randn(2, 3, 4, 6)  # [B, T_tok, N, d]
    target = torch.randint(0, 16, (2, 3, 4))
    loss = head.compute_loss(cond, target)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_compute_loss_rejects_float_targets():
    head = ClassifierTokenHead(cond_dim=6, codebook_size=16)
    cond = torch.randn(2, 3, 4, 6)
    target = torch.rand(2, 3, 4)  # float -> not valid token codes
    with pytest.raises(TypeError):
        head.compute_loss(cond, target)


def test_predict_and_metrics_shapes():
    head = ClassifierTokenHead(cond_dim=6, codebook_size=16)
    cond = torch.randn(2, 3, 4, 6)
    target = torch.randint(0, 16, (2, 3, 4))
    pred = head.predict(cond)
    assert pred.shape == (2, 3, 4)
    m = head.metrics(cond, target)
    assert "token_accuracy" in m and 0.0 <= float(m["token_accuracy"]) <= 1.0


def test_head_can_overfit_constant_target():
    torch.manual_seed(0)
    head = ClassifierTokenHead(cond_dim=6, codebook_size=8)
    cond = torch.randn(4, 2, 3, 6)
    target = torch.randint(0, 8, (4, 2, 3))
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    first = head.compute_loss(cond, target).item()
    for _ in range(2000):
        opt.zero_grad()
        loss = head.compute_loss(cond, target)
        loss.backward()
        opt.step()
    assert loss.item() < first
    assert float(head.metrics(cond, target)["token_accuracy"]) > 0.9
