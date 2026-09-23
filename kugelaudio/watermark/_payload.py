"""Coded customer-id payload — numpy mirror of the engine's codec.

The mark carries 2 raw bits per 133 ms chunk at a high error rate, so a
customer id is not sent directly: it selects a codeword spread across chunks
and is recovered by soft maximum-likelihood correlation over the codebook.
The codebook is generated from a fixed seed, so encoder and every detector
build the identical one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

CODEBOOK_SEED = 0xA17E
PAYLOAD_BITS = 12
CODEWORD_CHUNKS = 30
BITS_PER_CHUNK = 2
# RISK: genuine attribution recall drops if attacked field audio produces
# normalised margins below the synthetic/OOD calibration boundary. Exercise
# this threshold on the marked-audio robustness corpus before release.
PAYLOAD_MIN_MARGIN = 6.0
"""Minimum scale-normalised score gap required for attribution.

The threshold rejects the bounded random/OOD calibration cases while keeping
substantial headroom below an exact codeword's minimum observed margin (18).
Presence detection remains independent: an ambiguous payload reports no
customer id rather than turning the uncertainty into a false attribution.
"""


@dataclass(frozen=True)
class DecodedPayload:
    """Recovered id plus the evidence behind it."""

    customer_id: Optional[int]
    rotation: int
    margin: float
    normalized_margin: float
    score: float


class PayloadCodec:
    """Decodes a customer id from per-chunk bit logits."""

    def __init__(
        self, payload_bits: int = PAYLOAD_BITS, chunks: int = CODEWORD_CHUNKS
    ) -> None:
        import numpy as np

        self.payload_bits = payload_bits
        self.chunks = chunks
        self.slots = chunks * BITS_PER_CHUNK
        self.size = 1 << payload_bits
        rng = np.random.default_rng(CODEBOOK_SEED)
        book = rng.integers(0, 2, size=(self.size, self.slots), dtype=np.int8)
        self._bipolar = 2.0 * book.astype(np.float32) - 1.0

    def decode(
        self, logits: "np.ndarray", *, min_margin: float = 0.0
    ) -> DecodedPayload:
        """Recover the id, trying every chunk rotation.

        Blind by design: a detector handed a cropped file does not know where
        chunk 0 was, and soft correlation pays for the extra hypotheses many
        times over versus thresholding bits first.
        """
        import numpy as np

        flat = np.asarray(logits, dtype=np.float32).reshape(-1)
        if flat.size != self.slots:
            raise ValueError(f"expected {self.slots} logits, got {flat.size}")
        if not np.all(np.isfinite(flat)):
            raise ValueError("payload logits must all be finite")

        best_score = -np.inf
        runner_up = -np.inf
        best_value = 0
        best_rotation = 0
        for rotation in range(self.chunks):
            rolled = np.roll(flat, -rotation * BITS_PER_CHUNK)
            # RISK: numpy/Accelerate-specific ``matmul`` warnings previously
            # polluted otherwise finite scoring. ``dot`` keeps the equivalent
            # BLAS operation; parity is covered by the payload tests.
            scores = np.dot(self._bipolar, rolled)
            top = np.argpartition(scores, -2)[-2:]
            top = top[np.argsort(scores[top])][::-1]
            if float(scores[top[0]]) > best_score:
                runner_up = max(best_score, float(scores[top[1]]))
                best_score = float(scores[top[0]])
                best_value = int(top[0])
                best_rotation = rotation
            else:
                runner_up = max(runner_up, float(scores[top[0]]))

        margin = float(best_score - runner_up)
        evidence_scale = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))
        normalized_margin = margin / evidence_scale if evidence_scale > 0.0 else 0.0
        customer_id = best_value if normalized_margin >= min_margin else None
        return DecodedPayload(
            customer_id,
            best_rotation,
            margin,
            normalized_margin,
            best_score,
        )


def decode_with_sync(
    codec: PayloadCodec,
    logits_by_offset: "Mapping[int, np.ndarray]",
    *,
    min_margin: float = PAYLOAD_MIN_MARGIN,
) -> "tuple[DecodedPayload, int]":
    """Decode at the sample offset with the strongest correlation score.

    Whole-chunk rotations cannot recover a fractional-window shift introduced
    by MP3 or cropping. Callers therefore extract logits on a bounded grid of
    sample offsets and this function selects the best evidence without knowing
    the expected customer id.
    """
    if not logits_by_offset:
        raise ValueError("no payload offsets supplied")

    best: "tuple[DecodedPayload, int] | None" = None
    for offset, logits in logits_by_offset.items():
        attempt = codec.decode(logits, min_margin=0.0)
        if best is None or attempt.score > best[0].score:
            best = (attempt, offset)
    assert best is not None

    decoded, offset = best
    if decoded.normalized_margin < min_margin:
        decoded = DecodedPayload(
            None,
            decoded.rotation,
            decoded.margin,
            decoded.normalized_margin,
            decoded.score,
        )
    return decoded, offset


__all__ = [
    "CODEWORD_CHUNKS",
    "PAYLOAD_BITS",
    "PAYLOAD_MIN_MARGIN",
    "DecodedPayload",
    "PayloadCodec",
    "decode_with_sync",
]
