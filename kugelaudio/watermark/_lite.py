"""Torch-free forward pass for the KugelAudio watermark detector.

The detector is 37k parameters — six dilated Conv1d + GroupNorm + GELU layers
and two 1x1 heads — so it runs perfectly well on numpy alone. Implementing it
here rather than depending on torch is what keeps the `watermark` extra small
enough to install without thinking about it.

The weights ship beside this module as a plain ``.npz`` (~152 KB) with a
pinned SHA-256. Only the detector is published; the generator that mints marks
is not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .errors import WatermarkModelError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

ASSET_NAME = "watermark_sidelayer_detector_v1.npz"

ASSET_SHA256 = "985840e892f6f0885b320957ce8c5dbceff386bc3b0723ee55e53d1373a291cf"
"""Pinned digest of the shipped detector. A mismatch means the asset was
swapped or truncated, which would silently change every verdict, so loading
fails loudly instead."""

_GROUPS = 8
_EPS = 1e-5


def _asset_path() -> Path:
    return Path(__file__).parent / "assets" / ASSET_NAME


def _erf(x: "np.ndarray") -> "np.ndarray":
    """Abramowitz & Stegun 7.1.26 (max abs error 1.5e-7).

    numpy has no erf and pulling scipy for one function would defeat the point
    of this module. The approximation is ~4 orders of magnitude finer than the
    detector's own decision margins; parity against torch is asserted in the
    tests.
    """
    import numpy as np

    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * np.exp(-ax * ax))


def _gelu(x: "np.ndarray") -> "np.ndarray":
    """Exact GELU, matching torch.nn.GELU's default (erf form, not tanh)."""
    return 0.5 * x * (1.0 + _erf(x / 1.4142135623730951))


def _conv1d(
    x: "np.ndarray", weight: "np.ndarray", bias: "np.ndarray", dilation: int, pad: int
) -> "np.ndarray":
    """Batched dilated 1-D convolution: (B, Cin, T) -> (B, Cout, T).

    im2col + one matmul, so the heavy lifting lands in BLAS rather than a
    Python loop over taps.
    """
    import numpy as np

    batch, _, length = x.shape
    out_ch, in_ch, kernel = weight.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    # Gather the dilated taps for every output position.
    idx = np.arange(length)[:, None] + np.arange(kernel)[None, :] * dilation
    cols = x[:, :, idx]  # (B, Cin, T, K)
    cols = cols.transpose(0, 2, 1, 3).reshape(batch * length, in_ch * kernel)
    # RISK: numpy/Accelerate-specific ``matmul`` warnings previously polluted
    # otherwise finite inference. ``dot`` keeps the equivalent BLAS operation.
    out = np.dot(cols, weight.reshape(out_ch, in_ch * kernel).T) + bias
    return out.reshape(batch, length, out_ch).transpose(0, 2, 1)


def _group_norm(
    x: "np.ndarray", weight: "np.ndarray", bias: "np.ndarray", groups: int = _GROUPS
) -> "np.ndarray":
    """GroupNorm over (channels-in-group, time), as torch.nn.GroupNorm does."""
    import numpy as np

    batch, channels, length = x.shape
    grouped = x.reshape(batch, groups, channels // groups, length)
    mean = grouped.mean(axis=(2, 3), keepdims=True)
    var = grouped.var(axis=(2, 3), keepdims=True)
    normed = ((grouped - mean) / np.sqrt(var + _EPS)).reshape(batch, channels, length)
    return normed * weight[None, :, None] + bias[None, :, None]


def _sigmoid(x: "np.ndarray") -> "np.ndarray":
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@dataclass(frozen=True)
class LiteDetectorOutput:
    """Aggregated scores for one clip."""

    presence: float
    bits: "tuple[int, ...]"
    windows: int
    bit_logits: "Optional[np.ndarray]" = None
    """Per-window bit logits ``(windows, nbits)``, used by the payload codec."""


class LiteDetector:
    """The bundled 2-bit detector, running on numpy.

    Scores audio in fixed-length windows matching the training geometry and
    averages across them: GroupNorm normalises over the time axis, so feeding
    one long clip would not be equivalent to the windows the model was trained
    on.
    """

    def __init__(self, asset_path: Optional[Path] = None) -> None:
        import numpy as np

        path = asset_path or _asset_path()
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise WatermarkModelError(
                f"the bundled lite detector is missing at {path}"
            ) from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != ASSET_SHA256:
            raise WatermarkModelError(
                "the bundled lite detector failed its SHA-256 check: expected "
                f"{ASSET_SHA256}, got {digest}"
            )
        data = np.load(path)
        self._w = {k: data[k] for k in data.files if not k.startswith("meta_")}
        self.channels = int(data["meta_channels"])
        self.nbits = int(data["meta_nbits"])
        self.n_layers = int(data["meta_n_layers"])
        self.kernel_size = int(data["meta_kernel_size"])
        self.sample_rate = int(data["meta_sample_rate"])
        self.window_samples = int(data["meta_window_samples"])

    def forward(self, windows: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
        """(B, 1, T) float32 -> (presence_prob (B,), bit_logits (B, nbits))."""
        h = windows
        for layer in range(self.n_layers):
            dilation = 2**layer
            pad = (self.kernel_size - 1) * dilation // 2
            h = _conv1d(
                h,
                self._w[f"trunk.{layer}.0.weight"],
                self._w[f"trunk.{layer}.0.bias"],
                dilation,
                pad,
            )
            h = _group_norm(
                h, self._w[f"trunk.{layer}.1.weight"], self._w[f"trunk.{layer}.1.bias"]
            )
            h = _gelu(h)
        presence = _conv1d(h, self._w["presence_head.weight"], self._w["presence_head.bias"], 1, 0)
        bits = _conv1d(h, self._w["bits_head.weight"], self._w["bits_head.bias"], 1, 0)
        return _sigmoid(presence).mean(axis=(1, 2)), bits.mean(axis=-1)

    def score(self, samples_16k: "np.ndarray") -> LiteDetectorOutput:
        """Score a 1-D 16 kHz clip by averaging over whole windows."""
        import numpy as np

        n_windows = samples_16k.size // self.window_samples
        if n_windows == 0:
            raise WatermarkModelError(
                f"clip is shorter than one detector window "
                f"({self.window_samples} samples at {self.sample_rate} Hz)"
            )
        usable = samples_16k[: n_windows * self.window_samples]
        windows = usable.reshape(n_windows, 1, self.window_samples).astype(np.float32)
        presence, bit_logits = self.forward(windows)
        # Average LOGITS across windows before thresholding, mirroring how the
        # model integrates evidence within a window.
        mean_bits = bit_logits.mean(axis=0)
        return LiteDetectorOutput(
            presence=float(presence.mean()),
            bits=tuple(int(b) for b in (mean_bits > 0.0).astype(int)),
            windows=int(n_windows),
            bit_logits=bit_logits,
        )


__all__ = ["ASSET_SHA256", "LiteDetector", "LiteDetectorOutput"]
