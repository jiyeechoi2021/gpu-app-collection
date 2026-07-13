#!/usr/bin/env python3
"""Generate the Experiment A TENSOR sources from the immutable original."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


GENERATOR_VERSION = "task006-generator-v1"
SOURCE_RELATIVE_PATH = Path("tensor_benchmarks/TENSOR/tensorcore.cu")
SPEC_RELATIVE_PATH = Path("configs/tensor_warp_sweep.csv")

BLOCK_X_PATTERN = "   blockDim.x = 128;"
BLOCK_Y_PATTERN = "   blockDim.y = 4;"
GRID_X_PATTERN = (
    "   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / "
    "(WMMA_M * blockDim.x / 32);"
)
GRID_Y_PATTERN = (
    "   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / "
    "(WMMA_N * blockDim.y);"
)
EXPECTED_PATTERNS = (
    BLOCK_X_PATTERN,
    BLOCK_Y_PATTERN,
    GRID_X_PATTERN,
    GRID_Y_PATTERN,
)

EXPECTED_LAUNCHES = {
    "T0": (128, 4, 1, 16, 16, 1),
    "T1": (128, 2, 1, 16, 16, 1),
    "T2": (128, 1, 1, 16, 16, 1),
    "T3": (64, 1, 1, 16, 16, 1),
    "T4": (32, 1, 1, 16, 16, 1),
}

FIELD_NAMES = (
    "variant_id",
    "blockDim.x",
    "blockDim.y",
    "blockDim.z",
    "gridDim.x",
    "gridDim.y",
    "gridDim.z",
    "threads_per_block",
    "warps_per_block",
    "total_blocks",
    "total_launched_threads",
    "total_launched_warps",
    "requested_work_relative_to_T0",
)


@dataclass(frozen=True)
class Variant:
    variant_id: str
    block_x: int
    block_y: int
    block_z: int
    grid_x: int
    grid_y: int
    grid_z: int
    threads_per_block: int
    warps_per_block: int
    total_blocks: int
    total_launched_threads: int
    total_launched_warps: int
    requested_work_relative_to_t0: Decimal

    def manifest_record(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "block_dim": [self.block_x, self.block_y, self.block_z],
            "grid_dim": [self.grid_x, self.grid_y, self.grid_z],
            "threads_per_block": self.threads_per_block,
            "warps_per_block": self.warps_per_block,
            "total_blocks": self.total_blocks,
            "total_launched_threads": self.total_launched_threads,
            "total_launched_warps": self.total_launched_warps,
            "requested_work_relative_to_T0": format(
                self.requested_work_relative_to_t0, "f"
            ),
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def parse_int(row: dict[str, str], field: str, row_number: int) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: {field} must be an integer") from error


def load_variants(spec_path: Path) -> list[Variant]:
    with spec_path.open(newline="", encoding="utf-8") as spec_file:
        reader = csv.DictReader(spec_file)
        if tuple(reader.fieldnames or ()) != FIELD_NAMES:
            raise ValueError(
                "variant specification columns differ from the required schema: "
                f"expected {FIELD_NAMES}, found {tuple(reader.fieldnames or ())}"
            )
        variants: list[Variant] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                relative_work = Decimal(row["requested_work_relative_to_T0"])
            except (KeyError, InvalidOperation) as error:
                raise ValueError(
                    f"row {row_number}: requested_work_relative_to_T0 is invalid"
                ) from error
            variants.append(
                Variant(
                    variant_id=row["variant_id"],
                    block_x=parse_int(row, "blockDim.x", row_number),
                    block_y=parse_int(row, "blockDim.y", row_number),
                    block_z=parse_int(row, "blockDim.z", row_number),
                    grid_x=parse_int(row, "gridDim.x", row_number),
                    grid_y=parse_int(row, "gridDim.y", row_number),
                    grid_z=parse_int(row, "gridDim.z", row_number),
                    threads_per_block=parse_int(row, "threads_per_block", row_number),
                    warps_per_block=parse_int(row, "warps_per_block", row_number),
                    total_blocks=parse_int(row, "total_blocks", row_number),
                    total_launched_threads=parse_int(
                        row, "total_launched_threads", row_number
                    ),
                    total_launched_warps=parse_int(
                        row, "total_launched_warps", row_number
                    ),
                    requested_work_relative_to_t0=relative_work,
                )
            )
    if [variant.variant_id for variant in variants] != ["T0", "T1", "T2", "T3", "T4"]:
        raise ValueError("variant specification must contain T0-T4 exactly once and in order")
    return variants


def validate_variant_arithmetic(variants: Iterable[Variant]) -> None:
    variants = list(variants)
    t0_warps = variants[0].total_launched_warps
    for variant in variants:
        dimensions = (
            variant.block_x,
            variant.block_y,
            variant.block_z,
            variant.grid_x,
            variant.grid_y,
            variant.grid_z,
        )
        if dimensions != EXPECTED_LAUNCHES[variant.variant_id]:
            raise ValueError(
                f"{variant.variant_id}: launch dimensions differ from Experiment A: "
                f"expected {EXPECTED_LAUNCHES[variant.variant_id]}, found {dimensions}"
            )
        if any(value <= 0 for value in dimensions):
            raise ValueError(f"{variant.variant_id}: block/grid dimensions must be positive")
        threads = variant.block_x * variant.block_y * variant.block_z
        if threads != variant.threads_per_block:
            raise ValueError(f"{variant.variant_id}: threads_per_block arithmetic mismatch")
        if threads % 32 != 0 or threads > 1024:
            raise ValueError(
                f"{variant.variant_id}: block size must be a positive multiple of 32 "
                "and no greater than 1024"
            )
        if variant.block_x > 1024 or variant.block_y > 1024 or variant.block_z > 64:
            raise ValueError(f"{variant.variant_id}: block dimension exceeds CUDA limits")
        if variant.warps_per_block != threads // 32:
            raise ValueError(f"{variant.variant_id}: warps_per_block arithmetic mismatch")
        blocks = variant.grid_x * variant.grid_y * variant.grid_z
        if blocks != variant.total_blocks:
            raise ValueError(f"{variant.variant_id}: total_blocks arithmetic mismatch")
        if (variant.grid_x, variant.grid_y, variant.grid_z) != (16, 16, 1):
            raise ValueError(f"{variant.variant_id}: Experiment A grid must be (16,16,1)")
        if variant.total_launched_threads != blocks * threads:
            raise ValueError(
                f"{variant.variant_id}: total_launched_threads arithmetic mismatch"
            )
        if variant.total_launched_warps != blocks * variant.warps_per_block:
            raise ValueError(
                f"{variant.variant_id}: total_launched_warps arithmetic mismatch"
            )
        expected_relative = Decimal(variant.total_launched_warps) / Decimal(t0_warps)
        if variant.requested_work_relative_to_t0 != expected_relative:
            raise ValueError(
                f"{variant.variant_id}: requested_work_relative_to_T0 mismatch"
            )


def validate_original_patterns(original_text: str) -> None:
    for pattern in EXPECTED_PATTERNS:
        count = original_text.count(pattern)
        if count != 1:
            raise ValueError(
                f"expected original source pattern exactly once, found {count}: {pattern}"
            )


def generated_text(original_text: str, variant: Variant) -> str:
    if variant.variant_id == "T0":
        return original_text
    replacements = {
        BLOCK_X_PATTERN: f"   blockDim.x = {variant.block_x};",
        BLOCK_Y_PATTERN: f"   blockDim.y = {variant.block_y};",
        GRID_X_PATTERN: f"   gridDim.x = {variant.grid_x};",
        GRID_Y_PATTERN: f"   gridDim.y = {variant.grid_y};",
    }
    result = original_text
    for original, replacement in replacements.items():
        if result.count(original) != 1:
            raise ValueError(
                f"{variant.variant_id}: approved pattern was not found exactly once: {original}"
            )
        result = result.replace(original, replacement, 1)
    return result


def unified_diff(original_text: str, variant_text: str, variant_id: str) -> str:
    return "".join(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            variant_text.splitlines(keepends=True),
            fromfile="a/tensor_benchmarks/TENSOR/tensorcore.cu",
            tofile=f"b/generated/tensor_warp_sweep/sources/TENSOR_{variant_id}.cu",
            lineterm="\n",
        )
    )


def write_new_or_identical(path: Path, data: bytes) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise FileExistsError(
                f"refusing to overwrite non-identical generated artifact: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def generate(source_path: Path, spec_path: Path, output_dir: Path) -> dict[str, object]:
    source_directory = source_path.parent.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == source_directory or source_directory in resolved_output.parents:
        raise ValueError(
            "generated output directory must not be inside the original benchmark directory"
        )
    source_before = source_path.read_bytes()
    original_text = source_before.decode("utf-8")
    original_hash = sha256_bytes(source_before)
    makefile_path = source_path.with_name("Makefile")
    makefile_hash_before = sha256_file(makefile_path)

    variants = load_variants(spec_path)
    validate_variant_arithmetic(variants)
    validate_original_patterns(original_text)

    records: list[dict[str, object]] = []
    for variant in variants:
        record = variant.manifest_record()
        if variant.variant_id == "T0":
            record.update(
                {
                    "source_kind": "immutable_original",
                    "source_path": SOURCE_RELATIVE_PATH.as_posix(),
                    "source_sha256": original_hash,
                    "diff_path": None,
                    "diff_sha256": None,
                }
            )
        else:
            variant_text = generated_text(original_text, variant)
            variant_bytes = variant_text.encode("utf-8")
            diff_text = unified_diff(original_text, variant_text, variant.variant_id)
            diff_bytes = diff_text.encode("utf-8")
            source_output = output_dir / "sources" / f"TENSOR_{variant.variant_id}.cu"
            diff_output = output_dir / "diffs" / f"TENSOR_{variant.variant_id}.diff"
            write_new_or_identical(source_output, variant_bytes)
            write_new_or_identical(diff_output, diff_bytes)
            record.update(
                {
                    "source_kind": "mechanically_generated",
                    "source_path": f"sources/TENSOR_{variant.variant_id}.cu",
                    "source_sha256": sha256_bytes(variant_bytes),
                    "diff_path": f"diffs/TENSOR_{variant.variant_id}.diff",
                    "diff_sha256": sha256_bytes(diff_bytes),
                }
            )
        records.append(record)

    if source_path.read_bytes() != source_before:
        raise RuntimeError("original tensorcore.cu changed during generation")
    if sha256_file(makefile_path) != makefile_hash_before:
        raise RuntimeError("original TENSOR Makefile changed during generation")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "experiment": "fixed-grid coupled block-and-global warp-supply sweep",
        "original_source_path": SOURCE_RELATIVE_PATH.as_posix(),
        "original_source_sha256": original_hash,
        "original_makefile_sha256": makefile_hash_before,
        "variant_specification_path": SPEC_RELATIVE_PATH.as_posix(),
        "variant_specification_sha256": sha256_file(spec_path),
        "variants": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    write_new_or_identical(output_dir / "manifests" / "generation_manifest.json", manifest_bytes)
    return manifest


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Generate deterministic T1-T4 sources for TENSOR Experiment A."
    )
    parser.add_argument("--source", type=Path, default=root / SOURCE_RELATIVE_PATH)
    parser.add_argument("--variants", type=Path, default=root / SPEC_RELATIVE_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "generated/tensor_warp_sweep"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = generate(
            args.source.resolve(), args.variants.resolve(), args.output_dir.resolve()
        )
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        print(f"generation failed: {error}", file=sys.stderr)
        return 1
    print(
        "Generated T1-T4 and resolved T0 to the immutable original: "
        f"{args.output_dir.resolve()}"
    )
    print(f"Original source SHA-256: {manifest['original_source_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
