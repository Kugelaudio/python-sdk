"""Unit tests for turn-model metadata and integrity boundaries."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from kugelaudio.turn._bundle import download_bundle, materialize_bundle, verify_bundle
from kugelaudio.turn._schema import VariantIndex
from kugelaudio.turn.errors import TurnBundleError


def _write_bundle(root: Path) -> Path:
    files = {
        "audio.onnx": b"audio",
        "config.json": json.dumps(
            {
                "audio_encoder_name": "openai/whisper-small",
                "lm_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "sample_rate": 16000,
                "audio_window_s": 8.0,
                "encoder_frames_per_s": 50,
                "num_audio_tokens": 16,
                "audio_encoder_dim": 768,
                "whisper_input_frames": 800,
                "adapter_hidden_dim": 1024,
                "lm_hidden_dim": 896,
                "max_text_tokens": 128,
                "classes": ["complete", "incomplete", "backchannel", "wait"],
            }
        ).encode(),
        "feature_extractor/preprocessor_config.json": b"{}",
        "fusion.onnx": b"fusion",
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
    }
    declared = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        declared.append(
            {
                "path": relative,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "kugel-turn-onnx",
                "schema_version": 1,
                "source_checkpoint_sha256": "a" * 64,
                "audio_precision": "int4",
                "fusion_precision": "int8",
                "algorithm": "test",
                "input_feature_frames": 800,
                "files": declared,
            }
        )
    )
    return root


def test_verify_bundle_accepts_complete_checksums(tmp_path: Path) -> None:
    verified = verify_bundle(_write_bundle(tmp_path / "bundle"))

    assert verified.config.window_samples == 128000
    assert verified.manifest.audio_precision == "int4"


def test_verify_bundle_rejects_corrupt_weight(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle")
    (root / "fusion.onnx").write_bytes(b"broken")

    with pytest.raises(TurnBundleError, match="checksum mismatch"):
        verify_bundle(root)


def test_materialize_bundle_replaces_external_symlinks_with_real_files(
    tmp_path: Path,
) -> None:
    source = _write_bundle(tmp_path / "source")
    blob = tmp_path / "blob"
    blob.write_bytes((source / "audio.onnx").read_bytes())
    (source / "audio.onnx").unlink()
    (source / "audio.onnx").symlink_to(blob)

    materialized = materialize_bundle(source, tmp_path / "materialized")

    assert materialized.path == tmp_path / "materialized"
    assert materialized.path.joinpath("audio.onnx").is_symlink() is False
    assert materialized.path.joinpath("audio.onnx").read_bytes() == b"audio"


def test_variant_index_resolves_only_indexed_onnx_paths() -> None:
    index = VariantIndex.model_validate(
        {
            "schema_version": 1,
            "recommended": "onnx/recommended",
            "variants": [
                {"path": "onnx/recommended", "status": "recommended"},
                {"path": "float/source", "status": "source", "sha256": "b" * 64},
            ],
        }
    )

    assert index.resolve("recommended").path == "onnx/recommended"
    with pytest.raises(TurnBundleError, match="not a runtime bundle"):
        index.resolve("float/source")
    with pytest.raises(TurnBundleError, match="unknown"):
        index.resolve("onnx/missing")


def test_v2_variant_index_resolves_root_runtime_bundle() -> None:
    index = VariantIndex.model_validate(
        {
            "schema_version": 2,
            "recommended": "v2",
            "variants": [
                {
                    "path": "v2",
                    "status": "recommended",
                    "input_feature_frames": 1000,
                    "source_checkpoint_sha256": "c" * 64,
                }
            ],
        }
    )

    assert index.resolve("recommended").path == "v2"
    assert index.resolve("recommended").policy_path == "policy.json"
    assert index.resolve("recommended").input_feature_frames == 1000


def test_variant_index_resolves_per_variant_policy() -> None:
    index = VariantIndex.model_validate(
        {
            "schema_version": 2,
            "recommended": "models/w8a32",
            "variants": [
                {
                    "path": "models/w8a32",
                    "policy_path": "models/w8a32/policy.json",
                    "status": "recommended",
                },
                {
                    "path": "models/w8a8-qat",
                    "policy_path": "models/w8a8-qat/policy.json",
                    "status": "stable-supported-quality-exception",
                },
            ],
        }
    )

    assert index.resolve("recommended").policy_path == "models/w8a32/policy.json"
    assert index.resolve("models/w8a8-qat").policy_path == (
        "models/w8a8-qat/policy.json"
    )


def test_variant_index_rejects_unsafe_policy_path() -> None:
    with pytest.raises(ValueError, match="path must be safe and relative"):
        VariantIndex.model_validate(
            {
                "schema_version": 2,
                "recommended": "models/w8a32",
                "variants": [
                    {
                        "path": "models/w8a32",
                        "policy_path": "../policy.json",
                        "status": "recommended",
                    }
                ],
            }
        )


def test_download_bundle_uses_selected_variant_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    selected_path = "models/w8a8-qat"
    _write_bundle(snapshot / selected_path)
    selected_policy = snapshot / selected_path / "policy.json"
    selected_policy.write_text('{"selected": "w8a8"}')
    (snapshot / "policy.json").write_text('{"selected": "w8a32"}')
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "recommended": "models/w8a32",
                "variants": [
                    {
                        "path": "models/w8a32",
                        "policy_path": "models/w8a32/policy.json",
                        "status": "recommended",
                    },
                    {
                        "path": selected_path,
                        "policy_path": f"{selected_path}/policy.json",
                        "status": "stable-supported-quality-exception",
                    },
                ],
            }
        )
    )
    requested_patterns: list[str] = []

    def fake_hf_hub_download(*args: object, **kwargs: object) -> str:
        return str(variants_path)

    def fake_snapshot_download(*args: object, **kwargs: object) -> str:
        requested_patterns.extend(kwargs["allow_patterns"])
        return str(snapshot)

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = fake_hf_hub_download
    hub.snapshot_download = fake_snapshot_download
    errors = types.ModuleType("huggingface_hub.errors")
    for name in (
        "HfHubHTTPError",
        "LocalEntryNotFoundError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
    ):
        setattr(errors, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)

    downloaded = download_bundle(variant=selected_path)

    assert requested_patterns == [
        "models/w8a8-qat/**",
        "models/w8a8-qat/policy.json",
    ]
    assert downloaded.variant_path == selected_path
    assert downloaded.policy_path == selected_policy


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = "../audio.onnx"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(TurnBundleError, match="safe and relative"):
        verify_bundle(root)
