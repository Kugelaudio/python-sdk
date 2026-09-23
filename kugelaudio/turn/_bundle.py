"""Model acquisition and integrity verification for local turn detection."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ._schema import BundleManifest, TurnModelConfig, VariantIndex, load_schema
from .errors import TurnBundleError, TurnDependencyError, TurnModelDownloadError

DEFAULT_REPO_ID = "kugelaudio/turn-detection"
# Exact commit behind immutable tag v2.1.1. Keep the SDK default reproducible;
# channel branches are intended for explicit preview/operations workflows.
DEFAULT_REVISION = "21234773a703068a39f0d6a2db52726234747c03"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class VerifiedBundle:
    """A local bundle whose complete manifest has passed verification."""

    path: Path
    manifest: BundleManifest
    config: TurnModelConfig


@dataclass(frozen=True, slots=True)
class DownloadedBundle:
    """Verified model plus the metadata needed by the endpoint policy."""

    bundle: VerifiedBundle
    policy_path: Path
    variant_path: str
    revision: str


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Calculate SHA-256 without loading a model weight file into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise TurnBundleError(f"cannot read bundle file {path}: {exc}") from exc
    return digest.hexdigest()


def verify_bundle(bundle_dir: str | Path) -> VerifiedBundle:
    """Reject missing, corrupt, or incompatible model bundles."""
    root = Path(bundle_dir)
    manifest = load_schema(root / MANIFEST_FILENAME, BundleManifest)
    for item in manifest.files:
        path = root / item.path
        if not path.is_file():
            raise TurnBundleError(f"bundle file is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != item.size_bytes:
            raise TurnBundleError(
                f"bundle size mismatch for {item.path}: {actual_size} != {item.size_bytes}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != item.sha256:
            raise TurnBundleError(
                f"bundle checksum mismatch for {item.path}: {actual_sha} != {item.sha256}"
            )
    config = load_schema(root / "config.json", TurnModelConfig)
    if config.whisper_input_frames != manifest.input_feature_frames:
        raise TurnBundleError(
            "config/manifest frame mismatch: "
            f"{config.whisper_input_frames} != {manifest.input_feature_frames}"
        )
    return VerifiedBundle(path=root, manifest=manifest, config=config)


def materialize_bundle(
    source_dir: str | Path,
    destination_dir: str | Path,
) -> VerifiedBundle:
    """Create a real-path bundle for runtimes that reject HF cache symlinks."""
    source = verify_bundle(source_dir)
    destination = Path(destination_dir)
    if destination.exists():
        return verify_bundle(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-",
            dir=destination.parent,
        )
    )
    try:
        relative_paths = [MANIFEST_FILENAME]
        relative_paths.extend(item.path for item in source.manifest.files)
        for relative in relative_paths:
            source_path = (source.path / relative).resolve(strict=True)
            target_path = temporary / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.exists():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_bundle(destination)


def download_bundle(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    variant: str = "recommended",
    revision: str = DEFAULT_REVISION,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> DownloadedBundle:
    """Download only the selected variant, then verify every declared file."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.errors import (
            HfHubHTTPError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
        raise TurnDependencyError(
            'Turn detection needs optional dependencies; install "kugelaudio[turn-detection]".'
        ) from exc

    cache = str(cache_dir) if cache_dir is not None else None
    try:
        index_path = Path(
            hf_hub_download(
                repo_id,
                "variants.json",
                repo_type="model",
                revision=revision,
                token=token,
                cache_dir=cache,
                local_files_only=local_files_only,
            )
        )
        index = load_schema(index_path, VariantIndex)
        selected = index.resolve(variant)
        snapshot_path = Path(
            snapshot_download(
                repo_id,
                repo_type="model",
                revision=revision,
                token=token,
                cache_dir=cache,
                local_files_only=local_files_only,
                allow_patterns=[f"{selected.path}/**", selected.policy_path],
            )
        )
    except (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        raise TurnModelDownloadError(
            f"cannot acquire {repo_id}@{revision} variant {variant!r}: {exc}"
        ) from exc

    policy_path = snapshot_path / selected.policy_path
    if not policy_path.is_file():
        raise TurnBundleError(
            f"downloaded snapshot is missing policy metadata: {policy_path}"
        )
    source_bundle = snapshot_path / selected.path
    materialized_key = hashlib.sha256(
        f"{snapshot_path.resolve()}\0{selected.path}".encode()
    ).hexdigest()[:20]
    materialized = materialize_bundle(
        source_bundle,
        snapshot_path.parent.parent / "materialized" / materialized_key,
    )
    return DownloadedBundle(
        bundle=materialized,
        policy_path=policy_path,
        variant_path=selected.path,
        revision=revision,
    )
