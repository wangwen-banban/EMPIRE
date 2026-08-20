#!/usr/bin/env python3
"""Best-of-K EgoDex metrics for EXTERNAL (Being-H0) joint predictions.

Reuses side_errors/summarize/reconstruct_hand_joints from the original
compute_egodex_metrics.py byte-for-byte (imported), so formulas are identical.
GT comes from the reference eval npz shards (our pipeline's gt_actions +
current_states). Predictions are precomputed camera-frame joints in meters:
    pred_left_joints (n, K, 60, 21, 3), pred_right_joints, valid (n, K),
    global_indices (n,) indexing the reference window order (shard-concat).

--self_test M: instead of external predictions, rebuild joints from our own
sampled_actions for the first M windows and verify this script reproduces the
original per-window numbers (validates the joint-based path end to end).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

from empire.utils.data_utils import load_normalizer
from egodex_eval_utils import build_mano_layer, expand_entity_mask, reconstruct_hand_joints
from compute_egodex_metrics import side_errors, summarize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_dir", required=True, help="eval dir containing predictions_shard*.npz")
    p.add_argument("--pred_npz", nargs="*", default=None, help="Being-H0 driver output npz shards")
    p.add_argument("--output_json", required=True)
    p.add_argument("--statistics_path", default=os.environ.get("STATISTICS_PATH"))
    p.add_argument("--mano_model_path", default=os.environ.get("MANO_MODEL_PATH"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--anchor", action="store_true", help="translate pred so first-frame wrist matches GT (per side)")
    p.add_argument("--joint_perm", default=None, help="comma list len21: pred joint j comes from pred[perm[j]]")
    p.add_argument("--self_test", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.statistics_path or not args.mano_model_path:
        raise SystemExit("--statistics_path and --mano_model_path (or matching environment variables) are required.")
    shards = sorted(glob.glob(str(Path(args.ref_dir) / "predictions_shard*.npz")))
    assert shards, f"no shards in {args.ref_dir}"
    ga, cs, am, eid, fid, sa = [], [], [], [], [], []
    for s in shards:
        z = np.load(s, allow_pickle=True)
        ga.append(z["gt_actions"]); cs.append(z["current_states"]); am.append(z["action_masks"])
        eid.append(z["episode_ids"]); fid.append(z["frame_ids"])
        if args.self_test:
            sa.append(z["sampled_actions"])
    gt_actions = np.concatenate(ga); current_states = np.concatenate(cs)
    action_masks = np.concatenate(am)
    episode_ids = np.concatenate(eid); frame_ids = np.concatenate(fid)
    N = len(gt_actions)
    print(f"reference windows: {N}")

    normalizer = load_normalizer({"statistics_path": args.statistics_path})
    mano_layer = build_mano_layer(args.mano_model_path, args.device)

    perm = None
    if args.joint_perm:
        perm = np.array([int(x) for x in args.joint_perm.split(",")], dtype=np.int64)
        assert perm.shape == (21,)

    if args.self_test:
        sampled_actions = np.concatenate(sa)
        idxs = list(range(min(args.self_test, N)))
        K = sampled_actions.shape[1]
        get_pred = lambda i, k: reconstruct_hand_joints(sampled_actions[i, k], current_states[i], normalizer, mano_layer, args.device)
        valid = np.ones((N, K), dtype=bool)
    else:
        assert args.pred_npz, "--pred_npz required unless --self_test"
        zs = [np.load(p, allow_pickle=True) for p in args.pred_npz]
        K = zs[0]["pred_left_joints"].shape[1]
        T = zs[0]["pred_left_joints"].shape[2]
        PL = np.full((N, K, T, 21, 3), np.nan, dtype=np.float32)
        PR = np.full((N, K, T, 21, 3), np.nan, dtype=np.float32)
        valid = np.zeros((N, K), dtype=bool)
        for z in zs:
            g = z["global_indices"].astype(np.int64)
            PL[g] = z["pred_left_joints"]; PR[g] = z["pred_right_joints"]; valid[g] = z["valid"]
        cover = valid.any(axis=1)
        print(f"covered windows: {cover.sum()}/{N}; mean valid samples: {valid.sum(1)[cover].mean():.2f}")
        idxs = [i for i in range(N) if cover[i]]
        if args.limit:
            idxs = idxs[: args.limit]
        def get_pred(i, k):
            return {"left_joints": PL[i, k], "right_joints": PR[i, k]}

    records = []
    candidate_mpjpe = np.full((len(idxs), K), np.nan, dtype=np.float32)
    for row, i in enumerate(tqdm(idxs, desc="Being-H0 metrics")):
        mask = expand_entity_mask(action_masks[i])
        gt = reconstruct_hand_joints(gt_actions[i], current_states[i], normalizer, mano_layer, args.device)
        best_record, best_score, best_k = None, np.inf, 0
        for k in range(K):
            if not args.self_test and not valid[i, k]:
                continue
            pred = get_pred(i, k)
            pl = np.asarray(pred["left_joints"], dtype=np.float64).copy()
            pr = np.asarray(pred["right_joints"], dtype=np.float64).copy()
            if perm is not None:
                pl = pl[:, perm]; pr = pr[:, perm]
            if args.anchor:
                gl = np.asarray(gt["left_joints"]); gr = np.asarray(gt["right_joints"])
                pl += (gl[0, 0] - pl[0, 0]); pr += (gr[0, 0] - pr[0, 0])
            record = side_errors({"left_joints": pl, "right_joints": pr}, gt, mask)
            candidate_mpjpe[row, k] = record["mpjpe"]
            score = record["mpjpe"] if not np.isnan(record["mpjpe"]) else np.inf
            if best_record is None or score < best_score:
                best_score, best_record, best_k = score, record, k
        if best_record is None:
            continue
        rec = {"sample_index": int(i), "episode_id": str(episode_ids[i]), "frame_id": int(frame_ids[i]), "best_k": int(best_k)}
        rec.update(best_record)
        records.append(rec)

    summary = summarize(records)
    out = {"ref_dir": args.ref_dir, "pred_npz": args.pred_npz, "anchor": bool(args.anchor),
           "best_of_k": K, "n_windows": len(records), "summary": summary, "records": records}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as h:
        json.dump(out, h, indent=2)
    mm = summary
    print(f"OVERALL MPJPE {mm['overall_mpjpe_mm']:.2f} mm | FDE {mm['overall_fde_mm']:.2f} | "
          f"wrist {mm['overall_wrist_mpjpe_mm']:.2f} | finger-rel {mm['overall_finger_relative_mpjpe_mm']:.2f} | n={len(records)}")


if __name__ == "__main__":
    main()
