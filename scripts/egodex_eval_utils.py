"""Shared utilities for EgoDex quantitative and qualitative evaluation."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from empire.datasets.schema import load_episode_metadata, resolve_dataset_layout
from empire.utils.data_utils import recon_traj


HAND_DIM = 51
BETAS_DIM = 10
LEFT_STATE_SLICE = slice(0, 51)
RIGHT_STATE_PADDED_SLICE = slice(51, 102)
RIGHT_STATE_STATS_SLICE = slice(61, 112)

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
LEFT_AXIS_TRANSFORMATION_NP = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float32,
)


def expand_entity_mask(mask: np.ndarray) -> np.ndarray:
    """Return per-entity mask [T, 2] from padded/per-dim/per-entity masks."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask for one sample, got {mask.shape}")
    if mask.shape[1] <= 3:
        return mask[:, :2] > 0.5
    return np.stack([mask[:, 0] > 0.5, mask[:, HAND_DIM] > 0.5], axis=1)


def unnormalize_padded_state(norm_state: np.ndarray, normalizer) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover unpadded left/right hand states from a padded normalized state."""
    norm_state = np.asarray(norm_state, dtype=np.float32)
    state_mean = np.asarray(normalizer.state_mean, dtype=np.float32)
    state_std = np.asarray(normalizer.state_std, dtype=np.float32)

    left = norm_state[LEFT_STATE_SLICE] * (state_std[LEFT_STATE_SLICE] + 1e-7) + state_mean[LEFT_STATE_SLICE]
    right = norm_state[RIGHT_STATE_PADDED_SLICE] * (state_std[RIGHT_STATE_STATS_SLICE] + 1e-7) + state_mean[RIGHT_STATE_STATS_SLICE]

    # EgoDex conversion uses zero betas; the padded model state intentionally drops betas.
    beta_left = np.zeros(BETAS_DIM, dtype=np.float32)
    beta_right = np.zeros(BETAS_DIM, dtype=np.float32)
    return left.astype(np.float32), beta_left, right.astype(np.float32), beta_right


def unnormalize_actions(norm_actions: np.ndarray, normalizer, action_dim: int = 102) -> np.ndarray:
    norm_actions = np.asarray(norm_actions, dtype=np.float32)[..., :action_dim]
    action_mean = np.asarray(normalizer.action_mean, dtype=np.float32)[:action_dim]
    action_std = np.asarray(normalizer.action_std, dtype=np.float32)[:action_dim]
    return norm_actions * (action_std + 1e-7) + action_mean


def euler_traj_to_rotmat_traj(euler_traj: np.ndarray) -> np.ndarray:
    euler_traj = np.asarray(euler_traj, dtype=np.float32)
    t = euler_traj.shape[0]
    return R.from_euler("xyz", euler_traj.reshape(-1, 3)).as_matrix().reshape(t, 15, 3, 3)


def patch_legacy_deps():
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def build_manopth_layers(device: str):
    patch_legacy_deps()
    repo_value = os.environ.get("EGODEX_REPO")
    if not repo_value:
        raise RuntimeError("Set EGODEX_REPO when selecting the optional manopth backend")
    egodex_repo = Path(repo_value)
    sys.path.insert(0, str(egodex_repo / "manopth"))
    sys.path.insert(0, str(egodex_repo / "manopth" / "mano"))
    try:
        from manopth.manolayer import ManoLayer
        import manopth.rotproj as rotproj
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "3D joint metrics and rendering require either smplx or EgoDex manopth with chumpy. "
            "Use a Python environment that can import smplx, or set METRIC_PYTHON/VIS_PYTHON "
            "to the EgoDex environment after applying the legacy chumpy compatibility patches."
        ) from exc

    def batch_rotprojs_device_safe(batches_rotmats):
        proj_rotmats = []
        target_device = batches_rotmats.device
        for batch_rotmats in batches_rotmats:
            proj_batch_rotmats = []
            for rotmat in batch_rotmats:
                u, _, v = rotmat.cpu().svd()
                projected = torch.matmul(u, v.transpose(0, 1))
                if projected.det() < 0:
                    projected[:, 2] = -1 * projected[:, 2]
                proj_batch_rotmats.append(projected.to(target_device))
            proj_rotmats.append(torch.stack(proj_batch_rotmats))
        return torch.stack(proj_rotmats)

    rotproj.batch_rotprojs = batch_rotprojs_device_safe

    mano_root = os.environ.get("EGODEX_MANOPTH_MODEL_ROOT")
    if not mano_root:
        raise RuntimeError(
            "Set EGODEX_MANOPTH_MODEL_ROOT to separately licensed MANO assets "
            "when selecting the optional manopth backend"
        )
    kwargs = {
        "mano_root": mano_root,
        "use_pca": False,
        "flat_hand_mean": True,
        "center_idx": 0,
        "root_rot_mode": "axisang",
        "joint_rot_mode": "rotmat",
    }
    try:
        layers = {
            "backend": "manopth",
            "left": ManoLayer(side="left", **kwargs).to(device),
            "right": ManoLayer(side="right", **kwargs).to(device),
        }
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "3D joint metrics and rendering require either smplx or EgoDex manopth with chumpy. "
            "Use a Python environment that can import smplx, or set METRIC_PYTHON/VIS_PYTHON "
            "to an environment with manopth/chumpy."
        ) from exc
    layers["left"].eval()
    layers["right"].eval()
    return layers


def build_mano_layer(mano_model_path: str, device: str):
    patch_legacy_deps()
    if os.environ.get("EGODEX_MANO_BACKEND", "").lower() == "manopth":
        return build_manopth_layers(device)
    try:
        from empire.models.mano import create_mano_layer

        layers = {
            "backend": "smplx",
            "left": create_mano_layer(mano_model_path, is_right=False, device=device),
            "right": create_mano_layer(mano_model_path, is_right=True, device=device),
        }
    except ModuleNotFoundError as exc:
        if exc.name != "smplx":
            raise
        return build_manopth_layers(device)
    for key, layer in layers.items():
        if key == "backend":
            continue
        layer.eval()
    return layers


def layer_for_side(mano_layers, side: str):
    if isinstance(mano_layers, dict):
        return mano_layers[side]
    return mano_layers


def mano_backend(mano_layers) -> str:
    if isinstance(mano_layers, dict) and "backend" in mano_layers:
        return str(mano_layers["backend"])
    return "smplx"


def joints_from_traj(traj: np.ndarray, betas: np.ndarray, mano_layer, device: str, backend: str = "smplx", side: str = "right") -> np.ndarray:
    """Convert [T, 51] camera-space hand trajectory to [T, 21, 3] joints."""
    traj = np.asarray(traj, dtype=np.float32)
    t = traj.shape[0]
    transl = traj[:, :3]
    global_orient = R.from_euler("xyz", traj[:, 3:6]).as_matrix().astype(np.float32)
    hand_pose = euler_traj_to_rotmat_traj(traj[:, 6:51]).astype(np.float32)

    if backend == "manopth":
        hand_pose_t = torch.from_numpy(hand_pose).to(device)
        betas_t = torch.from_numpy(np.asarray(betas, dtype=np.float32)).to(device).unsqueeze(0).repeat(t, 1)
        trans_zero = torch.zeros((t, 3), dtype=torch.float32, device=device)
        root_eye = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 1, 3, 3).repeat(t, 1, 1, 1)
        pose_coeffs = torch.cat([root_eye, hand_pose_t], dim=1)
        with torch.no_grad():
            _, joints_mm = mano_layer(pose_coeffs, th_betas=betas_t, th_trans=trans_zero)
            joints_local = (joints_mm.detach().cpu().numpy() / 1000.0).astype(np.float32)
    else:
        beta_t = torch.from_numpy(np.asarray(betas, dtype=np.float32)).to(device).unsqueeze(0).repeat(t, 1)
        hand_pose_axis_angle = R.from_matrix(hand_pose.reshape(-1, 3, 3)).as_rotvec().astype(np.float32).reshape(t, 45)
        hand_pose_t = torch.from_numpy(hand_pose_axis_angle).to(device)
        global_identity = torch.zeros((t, 3), dtype=torch.float32, device=device)

        with torch.no_grad():
            out = mano_layer(
                betas=beta_t,
                hand_pose=hand_pose_t,
                global_orient=global_identity,
            )
            joints_local = out.joints.detach().cpu().numpy().astype(np.float32)

    joints_centered = joints_local - joints_local[:, 0:1]
    if side == "left":
        joints_centered = joints_centered @ LEFT_AXIS_TRANSFORMATION_NP.T
    joints = (global_orient @ joints_centered.transpose(0, 2, 1)).transpose(0, 2, 1)
    joints += transl[:, None, :]
    return joints.astype(np.float32)


def reconstruct_hand_joints(
    norm_actions: np.ndarray,
    norm_current_state: np.ndarray,
    normalizer,
    mano_layer,
    device: str,
) -> Dict[str, np.ndarray]:
    left_state, beta_left, right_state, beta_right = unnormalize_padded_state(norm_current_state, normalizer)
    actions = unnormalize_actions(norm_actions, normalizer, action_dim=102)

    left_traj = recon_traj(left_state, actions[:, :HAND_DIM])
    right_traj = recon_traj(right_state, actions[:, HAND_DIM:2 * HAND_DIM])
    return {
        "left_traj": left_traj,
        "right_traj": right_traj,
        "left_joints": joints_from_traj(left_traj[1:], beta_left, layer_for_side(mano_layer, "left"), device, mano_backend(mano_layer), "left"),
        "right_joints": joints_from_traj(right_traj[1:], beta_right, layer_for_side(mano_layer, "right"), device, mano_backend(mano_layer), "right"),
    }


def sample_joint_errors(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    norm_current_state: np.ndarray,
    action_mask: np.ndarray,
    normalizer,
    mano_layer,
    device: str,
) -> Dict[str, float]:
    pred = reconstruct_hand_joints(pred_actions, norm_current_state, normalizer, mano_layer, device)
    gt = reconstruct_hand_joints(gt_actions, norm_current_state, normalizer, mano_layer, device)
    mask = expand_entity_mask(action_mask)

    out: Dict[str, float] = {}
    total_error = 0.0
    total_count = 0
    fde_errors = []

    for side_idx, side in enumerate(("left", "right")):
        valid = mask[:, side_idx]
        if not np.any(valid):
            out[f"{side}_mpjpe"] = np.nan
            out[f"{side}_fde"] = np.nan
            out[f"{side}_frames"] = 0
            continue
        dist = np.linalg.norm(pred[f"{side}_joints"] - gt[f"{side}_joints"], axis=-1).mean(axis=-1)
        out[f"{side}_mpjpe"] = float(dist[valid].mean())
        out[f"{side}_frames"] = int(valid.sum())
        final_idx = np.flatnonzero(valid)[-1]
        out[f"{side}_fde"] = float(dist[final_idx])
        total_error += float(dist[valid].sum())
        total_count += int(valid.sum())
        fde_errors.append(float(dist[final_idx]))

    out["mpjpe"] = float(total_error / total_count) if total_count > 0 else np.nan
    out["fde"] = float(np.mean(fde_errors)) if fde_errors else np.nan
    out["frames"] = int(total_count)
    return out


def load_episode(data_root: Path, dataset_name: str, episode_id: str) -> dict:
    """Load an episode through the shared preferred/legacy layout resolver."""
    layout = resolve_dataset_layout(data_root, dataset_name)
    return load_episode_metadata(layout, episode_id)


def ego_video_path(video_root: Path, episode: dict) -> Path:
    video_name = str(episode["video_name"])
    direct = video_root / f"{video_name}.mp4"
    if direct.exists():
        return direct
    return video_root / video_name


def trajectory_indices(frame_id: int, steps: int, episode_len: int, frame_sample_step: float) -> np.ndarray:
    raw = frame_id + np.arange(steps, dtype=np.float64) * frame_sample_step
    return np.rint(raw).astype(np.int64).clip(0, episode_len - 1)


def camera_to_world(points_cam: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    cam_to_world = np.linalg.inv(world_to_cam)
    points_h = np.concatenate([points_cam, np.ones((*points_cam.shape[:-1], 1), dtype=points_cam.dtype)], axis=-1)
    return (cam_to_world @ points_h.reshape(-1, 4).T).T[:, :3].reshape(points_cam.shape)


def world_to_camera(points_world: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points_world, np.ones((*points_world.shape[:-1], 1), dtype=points_world.dtype)], axis=-1)
    return (world_to_cam @ points_h.reshape(-1, 4).T).T[:, :3].reshape(points_world.shape)


def project(points_cam: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-4
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = intrinsics[0, 0] * points_cam[valid, 0] / z[valid] + intrinsics[0, 2]
    uv[valid, 1] = intrinsics[1, 1] * points_cam[valid, 1] / z[valid] + intrinsics[1, 2]
    return uv, valid


def first_existing(paths) -> Optional[Path]:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None
