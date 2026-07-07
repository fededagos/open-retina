# openretina/modules/tokenizers/ntt_calm.py
"""Frozen adapter around a pre-trained ntt CALM-VQ discrete tokenizer.

This is the *only* place in openretina that touches ``ntt`` (a heavy, optional
dependency that requires Python >= 3.13). Every ``import ntt`` / ``import mlflow``
therefore lives inside a function or method so that this module imports fine even
when neither package is installed. The pure preprocessing (transpose + binarize +
the ``T % 100`` check) is factored out into :func:`binarize_and_transpose` so it can
be unit-tested without ntt.

The tokenizer turns 1 kHz binary spike trains into discrete token codes at 10 Hz
(``patch_length=100`` → 100 ms per token), emitting a single codebook index per
(time-token, neuron) pair.
"""

import torch

# URI schemes that must be resolved to a local file via mlflow before loading.
_MLFLOW_URI_PREFIXES = ("mlflow-artifacts:", "mlflow:", "runs:", "models:")

# Number of 1 kHz samples that the CALM-VQ tokenizer collapses into one token.
_SAMPLES_PER_TOKEN = 100


def binarize_and_transpose(spikes_bnt: torch.Tensor) -> torch.Tensor:
    """Turn spike *counts* ``[B, N, T]`` into binary tokenizer input ``[B, T, N]``.

    Pure torch, no ntt: transpose the neuron/time axes and binarize (``counts > 0``).
    The CALM-VQ tokenizer patches the time axis in blocks of 100 samples, so ``T``
    must be a multiple of 100 (100 ms per token at 1 kHz).

    Args:
        spikes_bnt: spike counts, shape ``[B, N, T]`` (batch, neurons, time @ 1 kHz).

    Returns:
        A ``LongTensor`` of shape ``[B, T, N]`` with values in ``{0, 1}``.

    Raises:
        ValueError: if ``spikes_bnt`` is not 3-D or ``T`` is not a multiple of 100.
    """
    if spikes_bnt.ndim != 3:
        raise ValueError(f"Expected spikes of shape [B, N, T] (3 dims); got shape {tuple(spikes_bnt.shape)}.")
    _b, _n, t = spikes_bnt.shape
    if t % _SAMPLES_PER_TOKEN != 0:
        raise ValueError(
            f"Time axis T={t} must be a multiple of {_SAMPLES_PER_TOKEN} "
            f"(100 ms/token at 1 kHz); got remainder {t % _SAMPLES_PER_TOKEN}."
        )
    spikes_btn = spikes_bnt.transpose(1, 2)  # [B, N, T] -> [B, T, N]
    return (spikes_btn > 0).long()


def _import_binned_spikes_input() -> type:
    """Lazily import ``ntt``'s ``BinnedSpikesInput`` container class.

    Wrapped in its own function so tests can monkeypatch this symbol with a fake
    without ntt being installed.
    """
    from ntt.core.components.types import BinnedSpikesInput

    return BinnedSpikesInput


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve an mlflow-artifacts URI to a local path, or pass a local path through.

    Downloading needs ``MLFLOW_TRACKING_URI`` / credentials in the environment; that
    is the caller's responsibility, not this module's.
    """
    if str(checkpoint_path).startswith(_MLFLOW_URI_PREFIXES):
        import mlflow.artifacts

        return mlflow.artifacts.download_artifacts(str(checkpoint_path))
    return str(checkpoint_path)


class FrozenNttTokenizer:
    """Frozen wrapper around a pre-trained ntt CALM-VQ tokenizer (a *teacher*).

    Deliberately **not** an :class:`torch.nn.Module`: a ``LightningModule`` that holds
    this as an attribute must keep the tokenizer out of its ``state_dict`` and out of
    its own ``.to()`` sweep. This wrapper therefore owns device alignment itself,
    moving the internal tokenizer to match the input on the first ``tokenize`` call
    (and again only if the device changes).
    """

    def __init__(self, tokenizer, sampling_rate: float = 1000.0):
        """Wrap an already-loaded, already-frozen ntt tokenizer.

        Args:
            tokenizer: the ntt tokenizer object (in ``.eval()`` with grads disabled).
            sampling_rate: input sampling rate in Hz (CALM-VQ was trained at 1 kHz).
        """
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate
        self._device: torch.device | None = None
        # Lazily-built caches for the response-space (rate) scoring path.
        self._samples_per_token: int | None = None
        self._decode_tables: dict[int, torch.Tensor] = {}  # n_tokens -> [n_tokens, vocab, spt]

    @property
    def vocab_size(self) -> int:
        """Number of codebook entries (2048 for the reference CALM-VQ checkpoint)."""
        return int(self.tokenizer.get_vocab_size())

    def to(self, device) -> "FrozenNttTokenizer":
        """Move the internal tokenizer to ``device`` (idempotent) and return ``self``."""
        device = torch.device(device)
        if self._device != device:
            self.tokenizer = self.tokenizer.to(device)
            self._device = device
        return self

    def tokenize(self, spikes_bnt: torch.Tensor) -> torch.Tensor:
        """Encode 1 kHz spike counts into discrete token codes.

        Args:
            spikes_bnt: spike *counts* at 1 kHz, shape ``[B, N, T]`` (float or int).
                ``T`` must be a multiple of 100.

        Returns:
            A ``LongTensor`` of codebook indices, shape ``[B, T_tok, N]`` with
            ``T_tok = T // 100``, values in ``[0, vocab_size)``.
        """
        spikes_btn = binarize_and_transpose(spikes_bnt)  # [B, T, N] long
        dev = spikes_btn.device
        self.to(dev)  # lazy, moves the tokenizer only when the device changes

        batch, time, num_units = spikes_btn.shape
        duration_sec = time / self.sampling_rate

        binned_spikes_input_cls = _import_binned_spikes_input()
        model_input = binned_spikes_input_cls(
            spike_matrix=spikes_btn.int(),  # [B, T, N]
            pad_and_gap_mask=torch.ones(batch, time, num_units, dtype=torch.int32, device=dev),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(num_units, dtype=torch.int32, device=dev).unsqueeze(0).expand(batch, num_units),
            unit_ids_pad_mask=torch.ones(batch, num_units, dtype=torch.int32, device=dev),
            duration_sec=torch.full((batch,), duration_sec, device=dev),
            session_id=("s",) * batch,
            subject_id=("s",) * batch,
            species=("m",) * batch,
            absolute_start=torch.zeros(batch, dtype=torch.float64, device=dev),
            absolute_end=torch.full((batch,), duration_sec, dtype=torch.float64, device=dev),
        )

        device_type = "cuda" if dev.type == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            tokens = self.tokenizer.encode(model_input)  # [B, T_tok, N, 1]
        return tokens.squeeze(-1).long()  # [B, T_tok, N]

    # -- Response-space (rate) decoding -------------------------------------------------
    #
    # For rate-correlation scoring we need the *firing rate* the tokenizer's decoder
    # reconstructs from token codes, not the discrete codes. Two entry points:
    #   * ``decode_tokens`` decodes concrete codes (used for the tokenizer-ceiling bound).
    #   * ``marginal_rate`` returns the closed-form expected rate over a per-token
    #     categorical posterior (Q=1 single-codebook), i.e. E[rate | stimulus].

    def _decode_raw(self, tokens_bdn1: torch.Tensor) -> torch.Tensor:
        """Run the frozen ntt decoder on ``[B, T_tok, N, 1]`` codes -> rate ``[B, T, N]``."""
        dev = tokens_bdn1.device
        self.to(dev)
        device_type = "cuda" if dev.type == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            return self.tokenizer.decode(tokens_bdn1.long())  # [B, T_tok * spt, N]

    @property
    def samples_per_token(self) -> int:
        """Output samples the decoder emits per token (100 for CALM-VQ @ 1 kHz).

        Probed once by decoding a single all-zero token; the length is deterministic
        (``patch_length``), so a length-1 probe is safe even though the decoder is
        contextual for its *values*.
        """
        if self._samples_per_token is None:
            probe = torch.zeros(1, 1, 1, 1, dtype=torch.long)
            self._samples_per_token = int(self._decode_raw(probe).shape[1])
        return self._samples_per_token

    def decode_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode concrete token codes ``[B, T_tok, N]`` to a rate ``[B, T_tok*spt, N]``."""
        return self._decode_raw(codes.unsqueeze(-1))

    def _decode_table(self, n_tokens: int, device: torch.device, chunk_size: int = 256) -> torch.Tensor:
        """Per-position decode table ``[n_tokens, vocab, spt]`` (cached per ``n_tokens``).

        For each token position ``pos`` and each codebook index ``k``, decodes a length
        ``n_tokens`` sequence with position ``pos`` set to ``k`` and every *other* position
        held at codebook index 0, then keeps position ``pos``'s output window. Mirrors
        ntt-downstream's ``build_ce_codebook_rates_table``.

        Two properties of CALM-VQ make this exact-for-what-it-decodes and cheap:
          * The decoder is per-neuron with shared weights, so the decoded window of a given
            code is identical across neurons -> the table has no neuron axis (built at N=1).
          * Decoding a full ``n_tokens`` sequence (rather than a lone token) keeps the
            bidirectional decoder near its trained sequence length.

        The residual bias (the "other positions" sit at real code 0, not a neutral element,
        so cross-position attention leaks an all-token-0 surround) is inherent to the closed
        -form marginal and is shared identically by every call, so scores stay internally
        consistent. Correlation, the only downstream use, is invariant to a global scale.
        """
        cached = self._decode_tables.get(n_tokens)
        if cached is not None:
            if cached.device != device:
                cached = cached.to(device)
                self._decode_tables[n_tokens] = cached
            return cached

        self.to(device)
        vocab = self.vocab_size
        spt: int | None = None
        position_tables: list[torch.Tensor] = []
        for pos in range(n_tokens):
            code_windows: list[torch.Tensor] = []
            for start in range(0, vocab, chunk_size):
                end = min(start + chunk_size, vocab)
                chunk = end - start
                idx = torch.zeros(chunk, n_tokens, 1, 1, dtype=torch.long, device=device)
                idx[:, pos, 0, 0] = torch.arange(start, end, device=device)
                decoded = self._decode_raw(idx)  # [chunk, n_tokens * spt, 1]
                if spt is None:
                    spt = decoded.shape[1] // n_tokens
                window = decoded[:, pos * spt : (pos + 1) * spt, 0]  # [chunk, spt]
                code_windows.append(window)
            position_tables.append(torch.cat(code_windows, dim=0))  # [vocab, spt]
        table = torch.stack(position_tables, dim=0)  # [n_tokens, vocab, spt]
        if self._samples_per_token is None:
            self._samples_per_token = int(spt)  # type: ignore[arg-type]
        self._decode_tables[n_tokens] = table
        return table

    def marginal_rate(self, posterior: torch.Tensor) -> torch.Tensor:
        """Closed-form expected rate over a per-token categorical posterior (Q=1).

        Args:
            posterior: per-token codebook probabilities ``[B, T_tok, N, K]`` (softmax of the
                head logits). ``K`` must equal :attr:`vocab_size`.

        Returns:
            Expected firing rate ``[B, T_tok * spt, N]`` at :attr:`sampling_rate`, computed as
            ``E[rate | stim] = sum_k posterior_k * decode(k)`` per token position.
        """
        b, t_tok, n, k = posterior.shape
        if k != self.vocab_size:
            raise ValueError(f"posterior last dim K={k} != tokenizer vocab_size {self.vocab_size}.")
        table = self._decode_table(t_tok, posterior.device)  # [T_tok, K, spt]
        # rate[b, pos, s, n] = sum_k posterior[b, pos, n, k] * table[pos, k, s]
        rate = torch.einsum("btnk,tks->btsn", posterior.float(), table.float())  # [B, T_tok, spt, N]
        return rate.reshape(b, t_tok * table.shape[-1], n)  # [B, T_tok * spt, N]


def load_frozen_ntt_tokenizer(
    checkpoint_path: str,
    map_location: str = "cpu",
    sampling_rate: float = 1000.0,
) -> FrozenNttTokenizer:
    """Load a pre-trained ntt CALM-VQ tokenizer from a checkpoint and freeze it.

    Resolves an ``mlflow-artifacts:`` (or ``mlflow:`` / ``runs:`` / ``models:``) URI to
    a local file if needed, loads the tokenizer via ntt, freezes it (``.eval()`` +
    ``requires_grad_(False)``), and wraps it in a :class:`FrozenNttTokenizer`.

    Args:
        checkpoint_path: local ``.ckpt`` path or an mlflow-artifacts URI. Resolving a
            URI needs ``MLFLOW_TRACKING_URI`` / credentials in the environment (the
            caller's responsibility).
        map_location: torch ``map_location`` for loading the checkpoint.
        sampling_rate: input sampling rate in Hz, forwarded to the wrapper.

    Returns:
        A frozen :class:`FrozenNttTokenizer`.
    """
    local_path = _resolve_checkpoint_path(checkpoint_path)

    from ntt.core.model import load_tokenizer_from_checkpoint

    # weights_only=False: torch>=2.6 defaults to True, but the checkpoint pickles an omegaconf
    # ListConfig in its hyper_parameters (same reason openretina main sets this on model load).
    # We trust our own tokenizer checkpoint.
    tokenizer = load_tokenizer_from_checkpoint(local_path, map_location=map_location, weights_only=False)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)
    return FrozenNttTokenizer(tokenizer, sampling_rate=sampling_rate)
