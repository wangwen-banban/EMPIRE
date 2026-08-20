#!/usr/bin/env python3
"""Merge sharded eval_mse.py .npz outputs back into full sample order."""

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge sharded EgoDex eval predictions.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    args = parser.parse_args()

    shard_data = [np.load(path, allow_pickle=True) for path in args.shards]
    if not shard_data:
        raise ValueError("No shard files provided")

    keys = set(shard_data[0].files)
    for data in shard_data[1:]:
        keys &= set(data.files)

    if "sample_positions" not in keys:
        raise ValueError("Shard files must contain sample_positions")

    merged = {}
    for key in sorted(keys):
        arrays = [data[key] for data in shard_data]
        if arrays[0].shape == ():
            merged[key] = arrays[0]
        else:
            merged[key] = np.concatenate(arrays, axis=0)

    order = np.argsort(merged["sample_positions"], kind="stable")
    for key, value in list(merged.items()):
        if getattr(value, "shape", ()) and value.shape[0] == len(order):
            merged[key] = value[order]

    expected = np.arange(len(order), dtype=merged["sample_positions"].dtype)
    if not np.array_equal(merged["sample_positions"], expected):
        raise ValueError(
            "Merged sample_positions are not contiguous after sorting: "
            f"got {merged['sample_positions'][:10]} ... {merged['sample_positions'][-10:]}"
        )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(args.output_path, **merged)
    print(f"Merged {len(args.shards)} shards into {args.output_path}")
    print(f"Total predictions: {merged['pred_actions'].shape[0]}")


if __name__ == "__main__":
    main()
