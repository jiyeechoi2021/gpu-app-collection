#!/usr/bin/env python3
"""Statically validate generated TENSOR Experiment A variants."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType


def load_generator(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tensor_variant_generator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def generation_artifact_bytes(root: Path) -> dict[str, bytes]:
    paths = list((root / "sources").glob("*"))
    paths += list((root / "diffs").glob("*"))
    paths.append(root / "manifests/generation_manifest.json")
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(paths)
        if path.is_file()
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Validate arithmetic, approved diffs, hashes, and determinism."
    )
    parser.add_argument(
        "--source", type=Path, default=root / "tensor_benchmarks/TENSOR/tensorcore.cu"
    )
    parser.add_argument(
        "--variants", type=Path, default=root / "configs/tensor_warp_sweep.csv"
    )
    parser.add_argument(
        "--generated-dir", type=Path, default=root / "generated/tensor_warp_sweep"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optionally preserve the static-validation result as JSON",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace, generator: ModuleType) -> dict[str, object]:
    source_path = args.source.resolve()
    makefile_path = source_path.with_name("Makefile")
    spec_path = args.variants.resolve()
    generated_dir = args.generated_dir.resolve()
    source_before = source_path.read_bytes()
    makefile_before = makefile_path.read_bytes()

    variants = generator.load_variants(spec_path)
    generator.validate_variant_arithmetic(variants)
    original_text = source_before.decode("utf-8")
    generator.validate_original_patterns(original_text)

    manifest_path = generated_dir / "manifests/generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["generator_version"] != generator.GENERATOR_VERSION:
        raise ValueError("generation manifest has the wrong generator version")
    if manifest["original_source_sha256"] != generator.sha256_bytes(source_before):
        raise ValueError("generation manifest original-source hash mismatch")
    if manifest["original_makefile_sha256"] != generator.sha256_bytes(makefile_before):
        raise ValueError("generation manifest original-Makefile hash mismatch")
    if manifest["variant_specification_sha256"] != generator.sha256_file(spec_path):
        raise ValueError("generation manifest variant-specification hash mismatch")

    records = {record["variant_id"]: record for record in manifest["variants"]}
    t0 = records["T0"]
    if t0["source_kind"] != "immutable_original":
        raise ValueError("T0 does not resolve to the immutable original")
    if t0["source_sha256"] != generator.sha256_bytes(source_before):
        raise ValueError("T0 source hash differs from the original")
    if t0["diff_path"] is not None:
        raise ValueError("T0 must not have a generated diff")

    for variant in variants[1:]:
        expected_text = generator.generated_text(original_text, variant)
        expected_diff = generator.unified_diff(original_text, expected_text, variant.variant_id)
        source_file = generated_dir / f"sources/TENSOR_{variant.variant_id}.cu"
        diff_file = generated_dir / f"diffs/TENSOR_{variant.variant_id}.diff"
        if source_file.read_text(encoding="utf-8") != expected_text:
            raise ValueError(f"{variant.variant_id}: generated source has an unapproved difference")
        if diff_file.read_text(encoding="utf-8") != expected_diff:
            raise ValueError(f"{variant.variant_id}: generated diff is not canonical")
        if records[variant.variant_id]["source_sha256"] != generator.sha256_file(source_file):
            raise ValueError(f"{variant.variant_id}: source hash mismatch")
        if records[variant.variant_id]["diff_sha256"] != generator.sha256_file(diff_file):
            raise ValueError(f"{variant.variant_id}: diff hash mismatch")

    with tempfile.TemporaryDirectory(prefix="tensor-variant-validation-") as first_dir_name:
        with tempfile.TemporaryDirectory(prefix="tensor-variant-validation-") as second_dir_name:
            first_dir = Path(first_dir_name)
            second_dir = Path(second_dir_name)
            generator.generate(source_path, spec_path, first_dir)
            generator.generate(source_path, spec_path, second_dir)
            first_files = directory_bytes(first_dir)
            second_files = directory_bytes(second_dir)
            if first_files != second_files:
                raise ValueError("two fresh generation runs produced different artifacts")
            if generation_artifact_bytes(generated_dir) != first_files:
                raise ValueError("checked generated directory differs from fresh generation")

    if source_path.read_bytes() != source_before:
        raise ValueError("original tensorcore.cu changed during validation")
    if makefile_path.read_bytes() != makefile_before:
        raise ValueError("original TENSOR Makefile changed during validation")

    return {
        "status": "passed",
        "checks": [
            "original source and Makefile unchanged during generation/validation",
            "T0 resolves to the unmodified original source",
            "T1-T4 contain only approved block/grid substitutions",
            "all grids are exactly (16,16,1)",
            "variant arithmetic and CUDA block limits are valid",
            "fresh repeated generation is byte-for-byte deterministic",
            "generated source, diff, specification, and manifest hashes match",
        ],
        "original_source_sha256": generator.sha256_bytes(source_before),
        "original_makefile_sha256": generator.sha256_bytes(makefile_before),
        "variant_specification_sha256": generator.sha256_file(spec_path),
        "generator_version": generator.GENERATOR_VERSION,
    }


def main() -> int:
    args = parse_args()
    try:
        generator = load_generator(Path(__file__).resolve().with_name(
            "generate_tensor_warp_variants.py"
        ))
        result = validate(args, generator)
    except (OSError, UnicodeError, ValueError, KeyError, RuntimeError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    result_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        generated_dir = args.generated_dir.resolve()
        if generated_dir != output_path.parent and generated_dir not in output_path.parents:
            print(
                "validation failed: --output-json must be inside --generated-dir",
                file=sys.stderr,
            )
            return 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_text, encoding="utf-8")
    print(result_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
