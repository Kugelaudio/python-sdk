"""Bounded-memory rational-ratio polyphase resampling, numpy-only.

The implementation evaluates only the FIR phase needed by each output sample.
It never materialises the zero-inserted signal at the least common sample rate,
which would turn one minute of 44.1 kHz audio into gigabytes of intermediates.
"""

from __future__ import annotations

from math import gcd
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

_FILTER_HALF_WIDTH = 16
"""Zero crossings kept either side of the sinc's centre. 16 puts the
transition band well below the watermark's own bandwidth; larger windows cost
time without changing the detector's verdict."""

_OUTPUT_BLOCK_SIZE = 8_192
"""Maximum output samples evaluated together.

The working set is ``block size * taps per phase`` and is independent of clip
duration and of the rational upsampling factor.
"""


def resample(samples: "np.ndarray", orig_rate: int, target_rate: int) -> "np.ndarray":
    """Resample a 1-D float array from ``orig_rate`` to ``target_rate``.

    Args:
        samples: 1-D float array.
        orig_rate: Source sample rate in Hz.
        target_rate: Destination sample rate in Hz.

    Returns:
        The resampled 1-D float32 array. Returned unchanged when the rates
        already match.
    """
    import numpy as np

    if orig_rate <= 0 or target_rate <= 0:
        raise ValueError(
            f"sample rates must be positive, got {orig_rate} -> {target_rate}"
        )
    if orig_rate == target_rate:
        return samples.astype(np.float32, copy=False)

    divisor = gcd(orig_rate, target_rate)
    up = target_rate // divisor
    down = orig_rate // divisor

    # Anti-aliasing cutoff is the lower of the two Nyquist limits, expressed
    # against the conceptual zero-inserted rate.
    max_rate = max(up, down)
    cutoff = 1.0 / max_rate
    half_len = _FILTER_HALF_WIDTH * max_rate
    n = np.arange(-half_len, half_len + 1, dtype=np.float64)
    kernel = np.sinc(n * cutoff) * np.blackman(2 * half_len + 1) * cutoff
    kernel = kernel / kernel.sum() * up

    # For output m, the conceptual high-rate position is m*down. Only source
    # samples n whose distance ``m*down - n*up`` lies inside the FIR matter.
    # Precompute the coefficients for each of the `up` possible phases, then
    # evaluate bounded blocks directly at the destination rate.
    tap_radius = (half_len + up - 1) // up
    tap_offsets = np.arange(-tap_radius, tap_radius + 1, dtype=np.int64)
    phases = np.arange(up, dtype=np.int64)[:, None]
    deltas = phases - tap_offsets[None, :] * up
    phase_valid = np.abs(deltas) <= half_len
    phase_coefficients = np.zeros(deltas.shape, dtype=np.float64)
    phase_coefficients[phase_valid] = kernel[deltas[phase_valid] + half_len]

    source = samples.astype(np.float32, copy=False)
    output_size = (source.size * up + down - 1) // down
    output = np.empty(output_size, dtype=np.float32)
    for start in range(0, output_size, _OUTPUT_BLOCK_SIZE):
        stop = min(start + _OUTPUT_BLOCK_SIZE, output_size)
        output_indices = np.arange(start, stop, dtype=np.int64)
        high_rate_positions = output_indices * down
        centres, output_phases = np.divmod(high_rate_positions, up)
        source_indices = centres[:, None] + tap_offsets[None, :]
        source_valid = (source_indices >= 0) & (source_indices < source.size)
        clipped_indices = np.clip(source_indices, 0, source.size - 1)
        coefficients = phase_coefficients[output_phases]
        output[start:stop] = np.sum(
            source[clipped_indices] * coefficients * source_valid,
            axis=1,
            dtype=np.float64,
        )
    return output


__all__ = ["resample"]
