import bisect
import copy
import json
import math
import os
import random
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from empire.datasets.augment_utils import (
    augmentation_func,
    center_crop_short_side,
    project_to_image_space,
)

from empire.datasets.interp_utils import interp_mano_state
from empire.datasets.schema import canonical_sample_id
from empire.datasets.video_utils import get_video_length, load_video_decord
from empire.datasets.dataset_utils import (
    compute_new_intrinsics_crop, 
    compute_new_intrinsics_resize, 
    calculate_fov,
    ActionFeature,
    StateFeature,
)
from empire.utils.data_utils import (
    read_dataset_statistics,
    GaussianNormalizer,
    resize_short_side_to_target,
)

class EpisodicDatasetCore(object):
    """Core dataset class for episodic hand manipulation data.
    
    Handles loading and processing of video frames, MANO hand parameters,
    and action sequences for hand-centric manipulation tasks.
    """
    def __init__(
        self,
        video_root,
        annotation_file, # episode_frame.npz
        label_folder,
        training_path=None,
        statistics_path=None,
        augmentation=True,
        flip_augmentation=True,
        set_none_ratio=0.0,
        action_type="angle",
        use_rel=False,
        use_obj=False,
        upsample_factor=1.0,
        target_image_width=224,
        clip_len=2000,
        state_mask_prob=0.1,
        action_past_window_size=0,
        action_future_window_size=15,
        image_past_window_size=0,
        image_future_window_size=0,
        rel_mode="step",
        load_images=True,
        use_obj_rel=False,
        use_cot=False,
        target_fps=12.0,
        source_fps=None,
        source_video_fps=30.0,
        filter_samples_by_fps=True,
        hand_feature_mode="full",
    ):
        self.video_root = video_root
        self.use_obj_rel = use_obj_rel
        self.use_cot = use_cot
        if annotation_file.endswith('.npz'):
            annotation_dict = np.load(annotation_file, allow_pickle=True)
        else:
            annotation_dict = np.load(annotation_file, allow_pickle=True).item()
        self.label_folder = label_folder
        self.index_frame_pair = annotation_dict['index_frame_pair'].copy()
        self.index_to_episode_id = annotation_dict['index_to_episode_id'].copy()

        if training_path is not None:
            self.training_idx = np.load(training_path, allow_pickle=True)
        else:
            self.training_idx = None

        if statistics_path is not None:
            self.data_statistics = read_dataset_statistics(statistics_path)

        self.global_data_statistics = None
        self.clip_len = clip_len  # Video clip length in frames
        self.augmentation = augmentation
        self.target_image_width = target_image_width
        self.flip_augmentation = flip_augmentation
        self.set_none_ratio = set_none_ratio
        self.action_type = action_type  # "angle" (Euler angles) or "keypoints" (3D joint positions)
        self.hand_feature_mode = self._normalize_hand_feature_mode(hand_feature_mode)
        self.use_rel = use_rel  # Whether to use relative delta as actions for hand poses (MANO poses)
        self.use_obj = use_obj  # Whether to include object motion prediction
        assert upsample_factor >= 1.0, "only support upsample_factor >= 1.0"
        self.upsample_factor = upsample_factor
        self.state_mask_prob = state_mask_prob  # Probability of masking state input
        self.target_fps = self._coerce_fps_value(target_fps, "target_fps")
        self.source_video_fps = self._coerce_fps_value(source_video_fps, "source_video_fps")
        self.source_fps = self._resolve_source_fps(source_fps)
        self.frame_sample_step = self._compute_frame_sample_step()
        self.sample_indices = self._build_sample_indices(filter_samples_by_fps)
        self.num_valid_frames = len(self.sample_indices)

        self.action_past_window_size=action_past_window_size
        self.action_future_window_size=action_future_window_size
        self.image_past_window_size=image_past_window_size
        self.image_future_window_size=image_future_window_size
        self.rel_mode=rel_mode
        self.load_images=load_images

    def __len__(self):
        return self.num_valid_frames

    @staticmethod
    def _normalize_hand_feature_mode(hand_feature_mode):
        mode = (hand_feature_mode or "full").lower()
        if mode in {"full", "all"}:
            return "full"
        if mode in {"wrist_6d_only", "wrist6d_only", "wrist_6d", "wrist6d", "wrist_only"}:
            return "wrist_6d_only"
        raise ValueError(
            f"Unknown hand_feature_mode={hand_feature_mode!r}; "
            "expected 'full' or 'wrist_6d_only'."
        )

    def _apply_hand_feature_mode(self, action, action_mask, state, state_mask):
        if self.hand_feature_mode == "full":
            return action, action_mask, state, state_mask
        if self.action_type != "angle":
            raise ValueError(
                "hand_feature_mode='wrist_6d_only' is only defined for angle actions."
            )

        action_keep = torch.zeros_like(action_mask, dtype=torch.bool)
        state_keep = torch.zeros_like(state_mask, dtype=torch.bool)

        for start, end in (ActionFeature.HUMAN_LEFT_6D, ActionFeature.HUMAN_RIGHT_6D):
            action_keep[:, start:end] = True
            state_keep[start:end] = True

        # Remove hand finger pose only. If an object stream is present, keep it
        # unchanged instead of silently dropping a non-hand condition.
        if self.use_obj:
            obj_start, obj_end = ActionFeature.OBJECT_6D
            action_keep[:, obj_start:obj_end] = True
            state_keep[obj_start:obj_end] = True

        action = action * action_keep.to(action.dtype)
        action_mask = action_mask & action_keep
        state = state * state_keep.to(state.dtype)
        state_mask = state_mask & state_keep
        return action, action_mask, state, state_mask

    @staticmethod
    def _coerce_fps_value(value, name):
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    @staticmethod
    def _snap_common_fps(fps):
        for common_fps in (1.0, 5.0, float(10), 12.0, 15.0, 24.0, 25.0, 30.0, 50.0, 60.0):
            if abs(fps - common_fps) <= max(0.05, common_fps * 0.02):
                return common_fps
        return float(fps)

    def _episode_id_from_data_id(self, data_id):
        corr = self.index_frame_pair[int(data_id)]
        episode_key = int(corr[0])
        return self.index_to_episode_id[episode_key]

    def _resolve_source_fps(self, source_fps):
        source_fps = self._coerce_fps_value(source_fps, "source_fps")
        if source_fps is not None:
            return source_fps
        if self.target_fps is None:
            return None

        inferred_fps = self._infer_source_fps_from_episodes()
        if inferred_fps is not None:
            return inferred_fps

        return self.target_fps

    def _infer_source_fps_from_episodes(self, max_probe_episodes=32):
        fps_candidates = []
        base_indices = self.training_idx if self.training_idx is not None else np.arange(len(self.index_frame_pair))
        seen_episode_ids = set()

        for data_id in np.asarray(base_indices[:max_probe_episodes * 4], dtype=np.int64):
            episode_id = self._episode_id_from_data_id(data_id)
            if episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(episode_id)

            epi_path = os.path.join(self.label_folder, episode_id + '.npy')
            if not os.path.exists(epi_path):
                continue
            epi = self._load_episode_npy(epi_path)

            for fps_key in ("fps", "annotation_fps", "sample_fps", "target_fps", "source_fps"):
                if fps_key in epi:
                    try:
                        fps_value = self._coerce_fps_value(epi[fps_key], fps_key)
                        if fps_value is not None:
                            fps_candidates.append(fps_value)
                            break
                    except (TypeError, ValueError):
                        pass
            else:
                decode_frame = np.asarray(epi.get("video_decode_frame", []), dtype=np.float64)
                if self.source_video_fps is not None and len(decode_frame) > 1:
                    diffs = np.diff(decode_frame)
                    positive_diffs = diffs[diffs > 0]
                    if len(positive_diffs) > 0:
                        mean_decode_step = float(np.mean(positive_diffs))
                        if mean_decode_step > 0:
                            fps_candidates.append(self.source_video_fps / mean_decode_step)

            if len(seen_episode_ids) >= max_probe_episodes:
                break

        if not fps_candidates:
            return None

        return self._snap_common_fps(float(np.median(fps_candidates)))

    def _compute_frame_sample_step(self):
        if self.target_fps is None or self.source_fps is None:
            return 1.0
        if self.target_fps > self.source_fps + 1e-6:
            raise ValueError(
                f"target_fps ({self.target_fps}) cannot exceed source_fps ({self.source_fps}); "
                "upsampling in the dataloader is not supported."
            )
        return max(1.0, float(self.source_fps) / float(self.target_fps))

    def _uses_temporal_sampling(self):
        return self.frame_sample_step > 1.0 + 1e-6

    def _build_sample_indices(self, filter_samples_by_fps):
        base_indices = self.training_idx if self.training_idx is not None else np.arange(len(self.index_frame_pair))
        base_indices = np.asarray(base_indices, dtype=np.int64)

        if not filter_samples_by_fps or not self._uses_temporal_sampling() or len(base_indices) == 0:
            return base_indices

        frames = np.asarray(self.index_frame_pair[base_indices, 1], dtype=np.float64)
        target_positions = np.rint(frames / self.frame_sample_step)
        aligned_frames = np.rint(target_positions * self.frame_sample_step)
        keep_mask = np.abs(frames - aligned_frames) < 0.5
        filtered_indices = base_indices[keep_mask]

        if len(filtered_indices) == 0:
            print(
                f"Warning: FPS sample filtering removed every sample "
                f"(source_fps={self.source_fps}, target_fps={self.target_fps}); using unfiltered samples."
            )
            return base_indices

        if len(filtered_indices) != len(base_indices):
            print(
                f"FPS sample filtering: kept {len(filtered_indices)}/{len(base_indices)} samples "
                f"for source_fps={self.source_fps}, target_fps={self.target_fps}."
            )
        return filtered_indices
    
    @staticmethod
    @lru_cache(maxsize=256)          # ~256 MB worst case if each npy ≈1 MB
    def _load_episode_npy(episode_path: str):
        """Load episode data from .npy file with caching.
        
        Uses LRU cache to keep up to 256 episodes in memory (~256 MB worst case).
        The cache automatically purges old entries when full.
        
        Args:
            episode_path: Path to the .npy file containing episode data
            
        Returns:
            Dictionary containing episode information
        """
        episode = np.load(episode_path, allow_pickle=True).item()
        if "plan" in episode and "cot" not in episode:
            episode["cot"] = episode["plan"]
        if "caption" in episode and "instruction" not in episode:
            episode["instruction"] = episode["caption"]
        return episode

    @staticmethod
    @lru_cache(maxsize=4096)
    def _get_video_num_frames(video_path: str) -> int:
        return int(get_video_length(video_path))

    def _clamp_decode_ids_to_video(self, video_path: str, decode_ids):
        decode_ids = np.asarray(decode_ids, dtype=np.int64)
        video_oob = np.zeros(decode_ids.shape, dtype=bool)

        num_frames = self._get_video_num_frames(video_path)
        if num_frames <= 0:
            raise ValueError(f"Video has no decodable frames: {video_path}")

        video_oob = (decode_ids < 0) | (decode_ids >= num_frames)
        if np.any(video_oob):
            decode_ids = np.clip(decode_ids, 0, num_frames - 1)

        return decode_ids.tolist(), video_oob

    def _load_or_cache_episode(self, episode_id):
        """
        Returns episode_result (raw dict) and the pre-extracted camera
        extrinsics (R_w2c, t_w2c).  No camera-space MANO tensors are cached.
        """
        root = self.label_folder
        epi_path = os.path.join(root, episode_id + '.npy')
        epi = self._load_episode_npy(epi_path)

        extr = epi['extrinsics']                         # world to cam, (T,4,4)
        R_w2c, t_w2c = extr[:, :3, :3], extr[:, :3, 3]

        return epi, R_w2c, t_w2c

    def _mat2euler(self, R_batch: np.ndarray) -> np.ndarray:
        """Batched XYZ-Euler conversion using SciPy."""
        flat = R_batch.reshape(-1, 3, 3)
        eul = R.from_matrix(flat).as_euler('xyz', degrees=False)
        return eul

    def _prepare_side_window(self, side_dict,
                            R_w2c, t_w2c,
                            idx_window, idx_anchor, # frame_id
                            *, anchor_frame=True, oob=None, start=None, end=None, upsample_factor=1.0,
                            next_idx=None, next_oob=None):

        T, W = len(side_dict['global_orient_worldspace']), len(idx_window)
        if oob is None:
            oob = np.zeros(W, dtype=bool)
        if next_idx is None:
            raw_next_idx = idx_window[-1] + 1
            next_oob = raw_next_idx > end
            next_idx = np.clip(raw_next_idx, start, end)
        if next_oob is None:
            next_oob = False

        idx_window_extend = np.append(idx_window, next_idx)
        oob_extend = np.append(oob, next_oob)

        kept_extend = side_dict['kept_frames'][idx_window_extend].astype(bool)

        R_mano_extend = side_dict['global_orient_worldspace'][idx_window_extend]
        t_mano_extend = side_dict['transl_worldspace'][idx_window_extend]
        hand_P_extend = side_dict['hand_pose'][idx_window_extend]
        joints_worldspace_extend = side_dict['joints_worldspace'][idx_window_extend]

        oob_indices = np.where(oob_extend)[0]
        if len(oob_indices) > 0:
            # If more than 0 frames are out of bounds, set kept to False
            kept_extend[oob_indices] = False

        if not np.all(kept_extend):
            identity = np.eye(3, dtype=hand_P_extend.dtype)
            identity_block = np.broadcast_to(identity, (hand_P_extend.shape[1], 3, 3))
            hand_P_extend[~kept_extend] = identity_block # directly copy the identity matrix for OOB parts
            R_mano_extend[~kept_extend] = identity

        # -------- camera-space (anchor camera) ----------------------------
        R_cam_extend  = R_w2c[idx_anchor] @ R_mano_extend # transform all mano global rotation matrices to the camera coordinates of frame_id
        t_cam_extend  = (R_w2c[idx_anchor] @ t_mano_extend[..., None])[..., 0] + t_w2c[idx_anchor] #transform all mano transl to the camera coordinates of frame_id


        # -------- finger Euler (batched) ----------------------------------
        pose_euler_extend  = R.from_matrix(hand_P_extend.reshape(-1,3,3))     \
                            .as_euler('xyz', degrees=False).reshape(-1,45)

        # -------- keypoints in mano space (batched) ----------------------------------
        joints_manospace_extend = (R_mano_extend.transpose(0, 2, 1) @ (joints_worldspace_extend.transpose(0, 2, 1) - t_mano_extend[..., None])).transpose(0,2,1)  # (W+1,21,3)

        if upsample_factor > 1:
            R_cam_extend, t_cam_extend, hand_P_extend, joints_manospace_extend, kept_extend = \
            interp_mano_state(R_cam_extend, t_cam_extend, hand_P_extend, joints_manospace_extend, kept_extend, upsample_factor, method="pchip")

            pose_euler_extend = R.from_matrix(hand_P_extend.reshape(-1,3,3))     \
                            .as_euler('xyz', degrees=False).reshape(-1,45)

            # set length back to W+1
            R_cam_extend = R_cam_extend[:W+1]
            t_cam_extend = t_cam_extend[:W+1]
            hand_P_extend = hand_P_extend[:W+1]
            pose_euler_extend = pose_euler_extend[:W+1]
            joints_manospace_extend = joints_manospace_extend[:W+1]
            kept_extend = kept_extend[:W+1]

        R_cam = R_cam_extend[:-1]
        t_cam = t_cam_extend[:-1]
        pose_euler = pose_euler_extend[:-1]
        hand_P = hand_P_extend[:-1]
        joints_manospace = joints_manospace_extend[:-1]
        kept = kept_extend[:-1]

        R_cam_next = R_cam_extend[1:]
        t_cam_next = t_cam_extend[1:]
        pose_euler_next = pose_euler_extend[1:]
        hand_P_next = hand_P_extend[1:]
        joints_manospace_next = joints_manospace_extend[1:]
        kept_next = kept_extend[1:]

        return dict(
            # current frame tensors
            R_cam=R_cam.astype(np.float32),
            t_cam=t_cam.astype(np.float32),
            pose_euler=pose_euler.astype(np.float32),
            hand_P=hand_P.astype(np.float32),
            joints_manospace = joints_manospace.astype(np.float32),
            kept=kept,

            # next-frame tensors (same length W)
            R_cam_next = R_cam_next.astype(np.float32),
            t_cam_next = t_cam_next.astype(np.float32),
            pose_euler_next = pose_euler_next.astype(np.float32),
            hand_P_next = hand_P_next.astype(np.float32),
            joints_manospace_next = joints_manospace_next.astype(np.float32),
            kept_next = kept_next,
        )

    def _prepare_object_window(self, obj_transforms,
                              R_w2c, t_w2c,
                              idx_window, idx_anchor,
                              *, oob=None, start=None, end=None, upsample_factor=1.0,
                              next_idx=None, next_oob=None):
        """Prepare object motion window from 4x4 transformation matrices.

        Args:
            obj_transforms: Object transformation matrices, shape (T, 4, 4)
            R_w2c: World to camera rotation matrices, shape (T, 3, 3)
            t_w2c: World to camera translation vectors, shape (T, 3)
            idx_window: Frame indices window
            idx_anchor: Anchor frame index
            oob: Out of bounds mask
            start: Start frame index
            end: End frame index
            upsample_factor: Upsampling factor

        Returns:
            Dictionary containing object pose in camera space
        """
        W = len(idx_window)
        if oob is None:
            oob = np.zeros(W, dtype=bool)
        if next_idx is None:
            raw_next_idx = idx_window[-1] + 1
            next_oob = raw_next_idx > end
            next_idx = np.clip(raw_next_idx, start, end)
        if next_oob is None:
            next_oob = False

        idx_window_extend = np.append(idx_window, next_idx)
        oob_extend = np.append(oob, next_oob)

        # Extract rotation and translation from 4x4 matrices
        R_obj_extend = obj_transforms[idx_window_extend, :3, :3]  # (W+1, 3, 3)
        t_obj_extend = obj_transforms[idx_window_extend, :3, 3]   # (W+1, 3)

        # Mark all frames as kept for object (objects are always visible)
        kept_extend = np.ones(W + 1, dtype=bool)

        # Handle out of bounds
        oob_indices = np.where(oob_extend)[0]
        if len(oob_indices) > 0:
            kept_extend[oob_indices] = False

        if not np.all(kept_extend):
            identity = np.eye(3, dtype=R_obj_extend.dtype)
            R_obj_extend[~kept_extend] = identity
            t_obj_extend[~kept_extend] = 0.0

        # Transform to camera space (anchor camera)
        R_cam_extend = R_w2c[idx_anchor] @ R_obj_extend
        t_cam_extend = (R_w2c[idx_anchor] @ t_obj_extend[..., None])[..., 0] + t_w2c[idx_anchor]

        # Upsampling if needed (simplified version - no joint interpolation)
        if upsample_factor > 1:
            # For upsampling, we can use linear interpolation on rotation and translation
            # This is a simplified version - more sophisticated interpolation could be used
            from empire.datasets.interp_utils import interp_mano_state

            # Create dummy joints and hand_pose for interpolation
            dummy_joints = np.zeros((W + 1, 1, 3), dtype=R_cam_extend.dtype)
            dummy_hand_P = np.broadcast_to(np.eye(3, dtype=R_cam_extend.dtype)[None, None, ...],
                                           (W + 1, 1, 3, 3))

            R_cam_extend, t_cam_extend, _, dummy_joints, kept_extend = \
                interp_mano_state(R_cam_extend, t_cam_extend, dummy_hand_P,
                                 dummy_joints, kept_extend, upsample_factor, method="pchip")

            # Trim back to W+1
            R_cam_extend = R_cam_extend[:W+1]
            t_cam_extend = t_cam_extend[:W+1]
            kept_extend = kept_extend[:W+1]

        # Split into current and next
        R_cam = R_cam_extend[:-1]
        t_cam = t_cam_extend[:-1]
        kept = kept_extend[:-1]

        R_cam_next = R_cam_extend[1:]
        t_cam_next = t_cam_extend[1:]
        kept_next = kept_extend[1:]

        return dict(
            R_cam=R_cam.astype(np.float32),
            t_cam=t_cam.astype(np.float32),
            kept=kept,
            R_cam_next=R_cam_next.astype(np.float32),
            t_cam_next=t_cam_next.astype(np.float32),
            kept_next=kept_next,
        )
    # ============================================================
    #  4.  Vectorised action-window constructor (ONE hand)
    # ============================================================
    def _make_action_window_vec(self, win, anchor_idx: int, *, rel_mode="step", action_type="angle"):
        """
        anchor_idx : the position of t0 inside the window (usually = past)
        rel_mode   : "step"   → Δ(t→t+1)
                    "anchor" → Δ(t→t0)
        action_type: "angle"  → Euler angles (xyz)
                    "keypoints " → root keypoints (21x3=63)
        """

        R_cur, t_cur  = win['R_cam'],        win['t_cam']
        R_nxt, t_nxt  = win['R_cam_next'],   win['t_cam_next']
        P_cur, P_nxt  = win['hand_P'],       win['hand_P_next'] # matrices
        pose_next     = win['pose_euler_next']
        kpoints_root_next = win['joints_manospace_next']
        kept, kept_n  = win['kept'],         win['kept_next']
        W = len(t_cur)

        # absolute pose of t+1
        if action_type == "keypoints":
            abs_next = kpoints_root_next.reshape(W, -1)
        elif action_type == "angle":
            abs_next = pose_next
        action_abs = np.concatenate(
            [t_nxt,
            self._mat2euler(R_nxt),
            abs_next],
            axis=-1).astype(np.float32)
        action_abs = action_abs.reshape(W, -1)

        # choose relative formulation
        if rel_mode == "step":
            t_rel = t_nxt - t_cur
            R_rel = R_nxt @ R_cur.transpose(0,2,1)
            P_rel = np.matmul(P_nxt, P_cur.transpose(0,1,3,2))
            valid = kept & kept_n

        elif rel_mode == "anchor":
            t_anchor  = t_cur[anchor_idx]
            R_anchor  = R_cur[anchor_idx]
            P_anchor  = P_cur[anchor_idx]

            # broadcast to all W rows
            t_rel = t_nxt - t_anchor
            R_rel = R_nxt @ R_anchor.T
            P_rel = np.matmul(P_nxt, P_anchor.transpose(0,2,1))
            valid = kept_n & kept[anchor_idx]

        else:
            raise ValueError('rel_mode must be "step" or "anchor"')

        pose_rel = R.from_matrix(P_rel.reshape(-1,3,3)) \
                    .as_euler('xyz',False).reshape(W,45)

        action_rel = np.concatenate(
            [t_rel,
            self._mat2euler(R_rel),
            pose_rel],
            axis=-1).astype(np.float32)

        action_abs[~valid] = 0.0
        action_rel[~valid] = 0.0
        return action_abs, action_rel, valid

    def _make_object_action_window_vec(self, win, anchor_idx: int, *, rel_mode="step"):
        """
        Create action window for object (6D: translation + rotation).

        anchor_idx : the position of t0 inside the window (usually = past)
        rel_mode   : "step"   → Δ(t→t+1)
                    "anchor" → Δ(t→t0)
        """
        R_cur, t_cur = win['R_cam'], win['t_cam']
        R_nxt, t_nxt = win['R_cam_next'], win['t_cam_next']
        kept, kept_n = win['kept'], win['kept_next']
        W = len(t_cur)

        # Absolute action: translation + rotation (Euler)
        action_abs = np.concatenate(
            [t_nxt, self._mat2euler(R_nxt)],
            axis=-1).astype(np.float32)  # (W, 6)

        # Relative action
        if rel_mode == "step":
            t_rel = t_nxt - t_cur
            R_rel = R_nxt @ R_cur.transpose(0, 2, 1)
            valid = kept & kept_n
        elif rel_mode == "anchor":
            t_anchor = t_cur[anchor_idx]
            R_anchor = R_cur[anchor_idx]
            t_rel = t_nxt - t_anchor
            R_rel = R_nxt @ R_anchor.T
            valid = kept_n & kept[anchor_idx]
        else:
            raise ValueError('rel_mode must be "step" or "anchor"')

        action_rel = np.concatenate(
            [t_rel, self._mat2euler(R_rel)],
            axis=-1).astype(np.float32)  # (W, 6)

        action_abs[~valid] = 0.0
        action_rel[~valid] = 0.0
        return action_abs, action_rel, valid

    def _indices_from_offsets(self, frame_id, offsets, start, end):
        raw_indices = frame_id + offsets.astype(np.float64) * self.frame_sample_step
        win = np.rint(raw_indices).astype(np.int64)
        oob = (win < start) | (win > end)
        return win.clip(start, end), oob

    def _window_indices(self, frame_id, past, future, start, end):
        """
        Returns:
            idx_clip : (W,) indices clipped to [0, T-1]
            oob      : (W,) bool — slots that were originally OOB
        """
        offsets = np.arange(-past, future + 1, dtype=np.float64)
        return self._indices_from_offsets(frame_id, offsets, start, end)

    def _next_index_after_window(self, frame_id, future, start, end):
        idx, oob = self._indices_from_offsets(
            frame_id,
            np.asarray([future + 1], dtype=np.float64),
            start,
            end,
        )
        return idx[0], oob[0]

    def _resolve_video_path(self, dataset_name: str = None, video_name: str = None, part_index: int = None) -> str:
        if dataset_name=='Ego4D':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name + '_part' + str(part_index+1) +'.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name +'.mp4')
            return video_path
        elif dataset_name=='EgoExo4D':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name +'_part' + str(part_index+1) +'.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name +'.mp4')
            return video_path
        elif dataset_name == 'epic':
            video_id = video_name.split('_')[0]
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name+ '_part' + str(part_index+1) + '.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name+ '.MP4')
            return video_path
        elif dataset_name == 'somethingsomethingv2':
            if self.clip_len is not None:
                video_path = os.path.join(self.video_root, video_name+ '_part' + str(part_index+1) + '.mp4')
            else:
                video_path = os.path.join(self.video_root, video_name+'.webm')
            return video_path
        elif dataset_name == 'arctic':
            # get the real action name
            video_paths = video_name.split('-')
            assert len(video_paths) == 3, f'Invalid video name {video_name} for arctic dataset'
            video_path = os.path.join(self.video_root, video_paths[0], video_paths[1].replace(' ', '_'), video_paths[2]) # only images under the path
            return video_path
        elif dataset_name == 'egodex':
            flat_candidates = [
                os.path.join(self.video_root, video_name + '.mp4'),
                os.path.join(self.video_root, video_name),
            ]
            if '_' in video_name:
                task_name, video_idx = video_name.rsplit('_', 1)
                flat_candidates.extend(
                    os.path.join(self.video_root, part_name, task_name, video_idx + '.mp4')
                    for part_name in ('test', 'part1', 'part2', 'part3', 'part4', 'part5')
                )
            for video_path in flat_candidates:
                if os.path.exists(video_path):
                    return video_path
            return flat_candidates[0]
        else:
            candidates = [
                os.path.join(self.video_root, video_name),
                os.path.join(self.video_root, video_name + '.mp4'),
            ]
            for video_path in candidates:
                if os.path.exists(video_path):
                    return video_path
            return candidates[0]

    def _mano_forward(self, betas, pose_m):
        """Runs MANO once and returns (vertices, joints) on CPU NumPy."""
        beta_t  = torch.tensor(betas).unsqueeze(0).float().cuda()
        pose_t  = torch.tensor(pose_m).unsqueeze(0).float().cuda()
        out     = mano(betas=beta_t, hand_pose=pose_t)       # no global_orient
        return out.vertices[0].cpu().numpy(), out.joints[0].cpu().numpy()

    # ------------------------------------------------------------
    #  Grab the (past + future + 1) frame window
    # ------------------------------------------------------------

    def _pack_state(self, R_cam, t_cam, pose_euler, idx):
        return np.concatenate([t_cam[idx],
                            self._mat2euler(R_cam[idx][None,...])[0],
                            pose_euler[idx]])

    def _grab_window_images(self,
                            episode_id: str,
                            epi: dict,
                            frame_id: int,
                            past: int,
                            future: int
                            ):
        """
        Returns
        -------
        images : (L, H, W, 3)  uint8   – raw RGB frames
        mask   : (L,) bool                True where real frame, False where pad
        """
        dataset_name = episode_id.split('_')[0]
        # video_path   = self._resolve_video_path(dataset_name, epi['video_name'])
        decode_table = epi['video_decode_frame']                      # (T,)
        T            = len(decode_table)
        frame_in_video = epi['video_decode_frame'][frame_id]

        # ---------- build padded window indices ---------------------------
        # Not support multiple images now
        idx_win, oob = self._window_indices(frame_id, past, future, 0, T-1)     # (W,) clipped to between 0 and T-1

        if self.clip_len is not None:
            part_idx = frame_in_video // self.clip_len # clip_len = 2000
            frame_in_part = frame_in_video % self.clip_len
            video_path = self._resolve_video_path(dataset_name, epi['video_name'], part_idx)
            decode_ids = [frame_in_part]
        else:
            video_path = self._resolve_video_path(dataset_name, epi['video_name'])
            decode_ids = decode_table[idx_win]

        decode_ids, video_oob = self._clamp_decode_ids_to_video(video_path, decode_ids)

        # ---------- read images --------------------
        # Retry mechanism: try up to 3 times to load video frames
        imgs = None
        for attempt in range(3):
            try:
                imgs, _ = load_video_decord(video_path, frame_index=decode_ids, rotation=False)
                break  # Success, exit the retry loop
            except Exception as e:
                # if attempt == 2:
                #     raise  # Raise the exception after 3 failed attempts
                print(f"Warning: failed to load video frames from {video_path} (attempt {attempt+1}/3): {e}")
                time.sleep(0.1)
        if imgs is None:
            raise RuntimeError(f"Failed to load video frames from {video_path} with frame_index={decode_ids}")

        images = np.stack(imgs, axis=0)           # (L,H,W,3) uint8
        image_oob = np.asarray(oob[:len(video_oob)], dtype=bool)
        mask   = ~(image_oob | video_oob)         # (L,) bool

        return images, mask

    def _find_matching_texts(self, text_list, frame_id):
        """Find the text annotation whose start_frame is closest to the given frame_id.

        Only returns text annotations that overlap with the current frame (start_frame <= frame_id < end_frame),
        and among those, returns the one with start_frame closest to frame_id.

        Args:
            text_list: List of tuples (text, (start_frame, end_frame))
            frame_id: Current frame ID to check

        Returns:
            matching_texts: List containing the single best matching text annotation (or empty if none)
            matching_ranges: List containing the corresponding time range (or empty if none)
            matching_indices: List containing the original index in text_list (or empty if none)

        Note:
            Uses half-open interval [start_frame, end_frame)
            Returns at most one text annotation - the one with start_frame closest to frame_id
        """
        matching_texts = []
        matching_ranges = []
        matching_indices = []

        best_text = None
        best_range = None
        best_idx = None
        min_distance = float('inf')

        for idx, (text, (start_frame, end_frame)) in enumerate(text_list):
            # Check if frame_id is in the half-open interval [start_frame, end_frame)
            if start_frame <= frame_id < end_frame:
                # Calculate distance from frame_id to start_frame
                distance = abs(frame_id - start_frame)

                # Keep the one with minimum distance to start_frame
                if distance < min_distance:
                    min_distance = distance
                    best_range = (start_frame, end_frame)
                    best_idx = idx

                    # text may be a list of annotations, we need to check each one
                    if isinstance(text, list):
                        best_text = text  # Keep the whole list for later selection
                    else:
                        best_text = text

        if best_text is not None:
            matching_texts.append(best_text)
            matching_ranges.append(best_range)
            matching_indices.append(best_idx)

        return matching_texts, matching_ranges, matching_indices

    def _random_select_text(
        self,
        text,
        text_rephrase,
        hand_type,
        clip_idx=None
    ):

        text_list = [text]
        if text_rephrase and isinstance(text_rephrase[hand_type][clip_idx][0], list):
            text_list.extend(text_rephrase[hand_type][clip_idx][0])

        text_selected = random.choice(text_list).strip()
        if not text_selected.endswith('.'):
            text_selected += '.'
        return text_selected

    def _normalize_main_type(self, main_type, text_clip=None):
        if main_type in {'left', 'right'}:
            return main_type

        if isinstance(text_clip, dict):
            left_has_text = len(text_clip.get('left', [])) > 0
            right_has_text = len(text_clip.get('right', [])) > 0
            if right_has_text and not left_has_text:
                return 'right'

        return 'left'

    def _build_instruction(
        self,
        main_type,
        text_clip,
        text_rephrase,
        idx_win,
        oob,
        epi_len, # T
        frame_id,
        action_past_window_size,
        action_future_window_size,
        cot=None
    ):

        main_type = self._normalize_main_type(main_type, text_clip)
        sub_type = 'right' if main_type == 'left' else 'left'

        # Build main text
        # text_clip[main_type][0]: ('Place the pink cup on the table.', (0, 26))
        # Find the text annotations for frame_id; there may be several, pick one at random
        main_text_list = text_clip[main_type]
        has_main_text = len(main_text_list) > 0

        main_text_selected = "None."
        main_win = (0, epi_len)  # Default to the full range if no text available

        # Initialize cot_text_selected to None
        cot_text_selected = None

        if cot is not None:
            # cot structure: List[Tuple(List[str], List[Tuple[int,int]])]
            # Find CoT text closest to the start of action window

            best_cot_text = None
            best_cot_range = None
            min_distance = float('inf')

            for cot_texts, cot_ranges in cot:
                # cot_texts: List[str] - multiple CoT descriptions
                # cot_ranges: Tuple[int, int] - (start, end) frame range
                start, end = cot_ranges

                # Calculate distance from frame_id to this CoT range
                if start <= frame_id <= end:
                    # frame_id is within this CoT range
                    distance = frame_id - start  # distance from frame_id to start of CoT range
                elif frame_id < start:
                    # frame_id is before this CoT range
                    distance = start - frame_id
                else:
                    # frame_id is after this CoT range
                    distance = frame_id - end

                # Select the CoT with minimum distance
                if distance < min_distance:
                    min_distance = distance
                    # Use the first CoT text (you can modify this if needed)
                    best_cot_text = random.choice(cot_texts) if len(cot_texts) > 0 else None
                    best_cot_range = (start, end)

            if best_cot_text is not None:
                cot_text_selected = best_cot_text

        if has_main_text:
                main_matching_texts, main_matching_ranges, main_matching_indices = self._find_matching_texts(main_text_list, frame_id)
                if len(main_matching_texts) > 0:
                    # Since we now return only the best match, use index 0
                    clip_idx = main_matching_indices[0]
                    main_win = main_matching_ranges[0]

                    main_text_selected = self._random_select_text(
                        main_matching_texts[0].strip(),
                        text_rephrase,
                        main_type,
                        clip_idx=clip_idx,
                    )


        # Build sub text
        sub_text_list = text_clip[sub_type]
        has_sub_text = len(sub_text_list) > 0

        sub_oob, sub_idx_win = oob, idx_win
        sub_text_selected = "None."
        sub_win = (0, epi_len)  # Default to the full range if no text available

        if has_sub_text:
            sub_matching_texts, sub_matching_ranges, sub_matching_indices = self._find_matching_texts(sub_text_list, frame_id)
            if len(sub_matching_texts) > 0:
                # Since we now return only the best match, use index 0
                clip_idx = sub_matching_indices[0]
                sub_win = sub_matching_ranges[0]
                sub_idx_win, sub_oob = self._window_indices(
                    frame_id,
                    action_past_window_size,
                    action_future_window_size, sub_win[0], sub_win[1]-1
                )     # (W,)

                sub_text_selected = self._random_select_text(
                    sub_matching_texts[0].strip(),
                    text_rephrase,
                    sub_type,
                    clip_idx=clip_idx,
                )

        # Assign left/right based on main_type
        is_main_left = (main_type == 'left')

        idx_win_left = idx_win if is_main_left else sub_idx_win
        idx_win_right = sub_idx_win if is_main_left else idx_win
        oob_left = oob if is_main_left else sub_oob
        oob_right = sub_oob if is_main_left else oob

        text_left = main_text_selected if is_main_left else sub_text_selected
        text_right = sub_text_selected if is_main_left else main_text_selected

        start_left = 0 if is_main_left else (sub_win[0] if has_sub_text else 0)
        start_right = (sub_win[0] if has_sub_text else 0) if is_main_left else 0
        end_left = epi_len - 1 if is_main_left else (sub_win[1] - 1 if has_sub_text else epi_len - 1)
        end_right = (sub_win[1] - 1 if has_sub_text else epi_len - 1) if is_main_left else epi_len - 1

        instruction = f"Left hand: {text_left} Right hand: {text_right}"


        return instruction, idx_win_left, oob_left, idx_win_right, oob_right, start_left, end_left, start_right, end_right, cot_text_selected

    def _get_2d_traj_cur_to_end(self, idx_frame, epi, intrinsics, hand_type, image_size):
        """Get the 2D trajectory of the hand palm from current frame to episode end.
        
        Args:
            idx_frame: Current frame index
            epi: Episode data dictionary
            intrinsics: Camera intrinsic matrix
            hand_type: 'left' or 'right' hand
            image_size: (H, W) tuple of image dimensions
            
        Returns:
            Normalized 2D palm trajectory in image space [0, 1]
        """
        H, W = image_size
        # intrinsics = epi['intrinsics'].copy()
        intrinsics = intrinsics.copy()
        # normalize intrinsics
        intrinsics[0] /= intrinsics[0,2]*2
        intrinsics[1] /= intrinsics[1,2]*2

        hand_joints_cur_to_end = epi[hand_type]['joints_worldspace'][idx_frame:] # (N, 21, 3)
        hand_palm_cur_to_end = np.mean(hand_joints_cur_to_end[:, [0,2,5,9,13,17], :], axis=1, keepdims=True) # (N, 1, 3)

        extrinsics = epi['extrinsics'].copy()
        extrinsics_cur = extrinsics[idx_frame] # world to cam
        R_world_to_cam = extrinsics_cur[None, :3, :3].repeat(len(hand_palm_cur_to_end), axis=0)
        t_world_to_cam = extrinsics_cur[None, :3, 3:].repeat(len(hand_palm_cur_to_end), axis=0)

        hand_palm_cur_to_end_cam = (R_world_to_cam @ hand_palm_cur_to_end.transpose(0, 2, 1) + t_world_to_cam).transpose(0, 2, 1)

        uv_palm_cur_to_end = project_to_image_space(hand_palm_cur_to_end_cam, intrinsics, (H, W)) # (N, M, 2)
        uv_palm_cur_to_end[..., 0] = np.clip(uv_palm_cur_to_end[..., 0], 0, W)
        uv_palm_cur_to_end[..., 1] = np.clip(uv_palm_cur_to_end[..., 1], 0, H)

        uv_palm_cur_to_end = uv_palm_cur_to_end.reshape(-1, 2)
        uv_palm_cur_to_end = uv_palm_cur_to_end.astype(np.float32)
        uv_palm_cur_to_end[:,0] /= W
        uv_palm_cur_to_end[:,1] /= H

        return uv_palm_cur_to_end

    def get_item_frame(
            self, episode_id, frame_id,
            action_past_window_size=0, 
            action_future_window_size=0,
            image_past_window_size=0, 
            image_future_window_size=0,
            rel_mode: str = "step",
            load_images: bool = True,
        ):
        """
        Vectorised dataloader.

        """
        # ------------------------------------------------------------------
        # 1. Load episode dict  +  extrinsics
        # ------------------------------------------------------------------
        epi, R_w2c, t_w2c = self._load_or_cache_episode(episode_id)
        T  = len(epi['extrinsics']) # 

        # ------------------------------------------------------------------
        # 2. Build frame-window indices
        # ------------------------------------------------------------------
        idx_win, oob  = self._window_indices(frame_id,
                                        action_past_window_size,
                                        action_future_window_size, 0, T-1)     # (W,)
        next_idx, next_oob = self._next_index_after_window(
            frame_id, action_future_window_size, 0, T - 1
        )
        W   = len(idx_win) # length of the chunk == 16

        # Check if object data exists in this episode
        has_object = self.use_obj and 'obj_rt' in epi

        obj_mesh_path = None
        if has_object and 'obj_mesh_path' in epi:
            obj_mesh_path = epi['obj_mesh_path']
        main_type = self._normalize_main_type(epi['anno_type'], epi.get('text'))
        sub_type = 'right' if main_type == 'left' else 'left'
        # ------------------------------------------------------------------
        # 3. Build instruction text
        # ------------------------------------------------------------------
        cot_text_selected = None
        if self.use_cot:
            instruction, idx_win_left, oob_left, idx_win_right, oob_right, \
            start_left, end_left, start_right, end_right, cot_text_selected = self._build_instruction(
                main_type = main_type,
                text_clip = epi['text'],
                text_rephrase = epi.get('text_rephrase'),
                idx_win = idx_win,
                oob = oob, # mask
                epi_len = T,
                frame_id = frame_id,
                action_past_window_size = action_past_window_size,
                action_future_window_size = action_future_window_size,
                cot = epi.get('cot', None) # List[Tuple(List[str], List[Tuple[int,int]])]] or None
            )
        else:
            instruction, idx_win_left, oob_left, idx_win_right, oob_right, \
            start_left, end_left, start_right, end_right, _ = self._build_instruction(
                main_type = main_type,
                text_clip = epi['text'],
                text_rephrase = epi.get('text_rephrase'),
                idx_win = idx_win,
                oob = oob, # mask
                epi_len = T,
                frame_id = frame_id,
                action_past_window_size = action_past_window_size,
                action_future_window_size = action_future_window_size,
            )

        next_idx_left, next_oob_left = self._next_index_after_window(
            frame_id, action_future_window_size, start_left, end_left
        )
        next_idx_right, next_oob_right = self._next_index_after_window(
            frame_id, action_future_window_size, start_right, end_right
        )
        

        # ------------------------------------------------------------------
        # 4. Vectorised actions  (left + right)
        # ------------------------------------------------------------------
        win_left  = self._prepare_side_window(
            epi['left'],  R_w2c, t_w2c, idx_win_left, frame_id, anchor_frame=True, 
            oob=oob_left, start=start_left, end=end_left, upsample_factor=self.upsample_factor,
            next_idx=next_idx_left, next_oob=next_oob_left
        )
        win_right = self._prepare_side_window(
            epi['right'], R_w2c, t_w2c, idx_win_right, frame_id, anchor_frame=True,
            oob=oob_right, start=start_right, end=end_right, upsample_factor=self.upsample_factor,
            next_idx=next_idx_right, next_oob=next_oob_right
        )

        # Object window (use same window as main hand)
        if has_object:
            win_obj = self._prepare_object_window(
                epi['obj_rt'], R_w2c, t_w2c, idx_win, frame_id,
                oob=oob, start=0, end=T-1, upsample_factor=self.upsample_factor,
                next_idx=next_idx, next_oob=next_oob
            )
        else:
            # If no object data, create dummy window
            W = len(idx_win)
            win_obj = {
                'R_cam': np.zeros((W, 3, 3), dtype=np.float32),
                't_cam': np.zeros((W, 3), dtype=np.float32),
                'kept': np.zeros(W, dtype=bool),
                'R_cam_next': np.zeros((W, 3, 3), dtype=np.float32),
                't_cam_next': np.zeros((W, 3), dtype=np.float32),
                'kept_next': np.zeros(W, dtype=bool),
            }

        idx_center = action_past_window_size          # local index of t0 in window

        # rel_mode: "step"  or  "anchor" / action_type: "angle" or "keypoints"
        # step: relative to previous frame, anchor: relative to t0
        abs_L, rel_L, msk_L = self._make_action_window_vec(
            win_left,  anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
        )

        abs_R, rel_R, msk_R = self._make_action_window_vec(
            win_right, anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
        )

        # Decide whether to include object based on has_object flag
        if has_object:
            abs_obj, rel_obj, msk_obj = self._make_object_action_window_vec(
                win_obj, anchor_idx=idx_center, rel_mode=rel_mode
            )
            action_abs = np.concatenate([abs_L, abs_R, abs_obj], axis=1)   # (W, 108) = 51+51+6
            action_rel = np.concatenate([rel_L, rel_R, rel_obj], axis=1)   # (W, 108)
            action_mask = np.stack([msk_L, msk_R, msk_obj], axis=1)        # (W, 3)

        else:
            action_abs = np.concatenate([abs_L, abs_R], axis=1)   # (W, 102) = 51+51
            action_rel = np.concatenate([rel_L, rel_R], axis=1)   # (W, 102)
            action_mask = np.stack([msk_L, msk_R], axis=1)        # (W, 2)

        cur_L = self._pack_state(win_left['R_cam'],
                    win_left['t_cam'],
                    win_left['pose_euler'] if self.action_type=='angle' else win_left['joints_manospace'].reshape(W, -1),
                    idx_center)

        cur_R = self._pack_state(win_right['R_cam'],
                            win_right['t_cam'],
                            win_right['pose_euler'] if self.action_type=='angle' else win_right['joints_manospace'].reshape(W, -1),
                            idx_center)


        betas_L = epi['left']['beta']
        betas_R = epi['right']['beta']

        # Decide whether to include object state based on has_object flag
        if has_object:
            # Object current state (6D: rotation + translation)
            cur_obj = np.concatenate([
                win_obj['t_cam'][idx_center],
                self._mat2euler(win_obj['R_cam'][idx_center][None, ...])[0]
            ])
            current_state = np.concatenate([cur_L, betas_L, cur_R, betas_R, cur_obj])  # 128 = 61+61+6
            current_state_mask = np.array([win_left['kept'][idx_center], win_right['kept'][idx_center], win_obj['kept'][idx_center]])
        else:
            current_state = np.concatenate([cur_L, betas_L, cur_R, betas_R])  # 122 = 61+61
            current_state_mask = np.array([win_left['kept'][idx_center], win_right['kept'][idx_center]])

        # ------------------------------------------------------------------
        # 5. RGB window
        # ------------------------------------------------------------------
        if load_images:
            image_list, image_mask = self._grab_window_images(
                episode_id, epi,
                frame_id,
                image_past_window_size,
                image_future_window_size
            )
            H = image_list[0].shape[0]
            W = image_list[0].shape[1]
        else:
            image_list = None
            image_mask = None
            H, W = epi['intrinsics'][1,2]*2, epi['intrinsics'][0,2]*2
        # ------------------------------------------------------------------
        # 6. Calculate New_intrinsics
        # ------------------------------------------------------------------
        dataset_name = episode_id.split('_')[0]
        intrinsics = epi['intrinsics']

        if dataset_name == 'EgoExo4D':
            # For EgoExo4D, the fisheye camera images contain black borders after undistortion.
            # We remove these borders using a center crop. Specifically, the video frames are
            # first resized from 1408 to 448, and then center-cropped to 256.

            new_intrinsics = compute_new_intrinsics_crop(intrinsics, 1408, 256/448*1408, H)
            
        else:
            new_intrinsics = compute_new_intrinsics_resize(intrinsics, (H, W))

        # ------------------------------------------------------------------
        # 7. Do augmentation
        # ------------------------------------------------------------------
        if self.augmentation:
            try:
                # randomly sample aspect ratio for augmentation
                aspect_ratio = np.exp(random.uniform(np.log(1.0), np.log(2.0)))
                target_size = (int(self.target_image_width * aspect_ratio), self.target_image_width)  # (W, H)
                augment_params = {
                    'tgt_aspect': aspect_ratio, 
                    'flip_augmentation': self.flip_augmentation, 
                    'set_none_ratio': self.set_none_ratio,
                }

                uv_traj = self._get_2d_traj_cur_to_end(frame_id, epi, new_intrinsics, main_type, (H, W))
                image_list, new_intrinsics, (action_abs, action_rel, action_mask), \
                (current_state, current_state_mask), instruction = \
                    augmentation_func(
                        image = image_list, 
                        intrinsics = new_intrinsics,
                        actions = (action_abs, action_rel, action_mask),
                        states = (current_state, current_state_mask),
                        captions = instruction,
                        uv_traj = uv_traj,
                        target_size = target_size,
                        augment_params = augment_params,
                        sub_type = sub_type,
                    )

            except Exception as e:
                print(f"Warning: Augmentation failed for episode {episode_id}, frame {frame_id}: {e}. Do center crop only")
                import traceback
                print(f"Warning: Augmentation failed for episode {episode_id}, frame {frame_id}")
                print(f"Exception: {type(e).__name__}: {e}")
                print(f"Traceback:\n{traceback.format_exc()}")
                image_list = center_crop_short_side(image_list[0])[None, ...]
                new_intrinsics[0][2] = 0.5 * image_list[0].shape[1]  # update the principal point
                new_intrinsics[1][2] = 0.5 * image_list[0].shape[0]  # update the principal point

            if random.random() < self.state_mask_prob:
                # Mask state based on has_object flag to maintain consistent dimensions
                if has_object:
                    current_state_mask = np.array([False, False, False])
                else:
                    current_state_mask = np.array([False, False])
                current_state[:] = 0.0

        fov = calculate_fov( 2 * new_intrinsics[1][2], 2 * new_intrinsics[0][2], new_intrinsics)

        if self.use_rel:
            action_list = action_rel
        else:
            # use abs action for hand pose only
            # When has_object: shape is (W, 108) = 51(L) + 51(R) + 6(obj)
            # When no object:  shape is (W, 102) = 51(L) + 51(R)
            if has_object:
                rel_L = action_rel[:, :51]
                rel_R = action_rel[:, 51:102]
                abs_L = action_abs[:, :51]
                abs_R = action_abs[:, 51:102]
                rel_obj = action_rel[:, 102:]
                abs_obj = action_abs[:, 102:]
                if self.use_obj_rel:
                    action_list = np.concatenate([rel_L[:, :6], abs_L[:, 6:], rel_R[:, :6], abs_R[:, 6:], rel_obj], axis=1)
                else:
                    action_list = np.concatenate([rel_L[:, :6], abs_L[:, 6:], rel_R[:, :6], abs_R[:, 6:], abs_obj], axis=1)
            else:
                rel_L = action_rel[:, :action_rel.shape[1]//2]
                rel_R = action_rel[:, action_rel.shape[1]//2:]
                abs_L = action_abs[:, :action_abs.shape[1]//2]
                abs_R = action_abs[:, action_abs.shape[1]//2:]
                action_list = np.concatenate([rel_L[:, :6], abs_L[:, 6:], rel_R[:, :6], abs_R[:, 6:]], axis=1)
                

        # ------------------------------------------------------------------
        # 8. Return to caller
        # ------------------------------------------------------------------

        result_dict = dict(
            instruction             = instruction,
            action_list             = action_list,          # (W,2*51) float32
            action_mask             = action_mask,          # (W,2)   bool or (W,3) bool
            current_state           = current_state,        # (2*61,)  float32 or (2*61+6,) float32
            current_state_mask      = current_state_mask,   # (2,) bool
            fov                     = fov,                  # (2,) float32
            intrinsics              = new_intrinsics,       # (3,3) float32
        )
        # Only add obj_mesh_path if it's not None (to avoid collate issues)
        if obj_mesh_path is not None:
            result_dict['obj_mesh_path'] = obj_mesh_path
        if cot_text_selected is not None:
            result_dict['cot'] = cot_text_selected

        if image_list is not None:
            result_dict['image_list'] = image_list          # (W,H,W,3) uint8
        if image_mask is not None:
            result_dict['image_mask'] = image_mask          # (W,) bool

        return result_dict

    def set_global_data_statistics(self, global_data_statistics):
        self.global_data_statistics = global_data_statistics
        if not hasattr(self, 'gaussian_normalizer'):
            self.gaussian_normalizer = GaussianNormalizer(self.global_data_statistics)

    def transform_trajectory(
        self,
        sample_dict: dict = None,
        normalization: bool = True,
    ):
        """Pad action and state dimensions to match a unified size."""
        action_np = sample_dict["action_list"]
        state_np = sample_dict["current_state"]

        action_dim = action_np.shape[1] # 108 (51+51+6)
        state_dim = state_np.shape[0] # 128 (61+61+6)
        if normalization:
            # Normalize left and right hand actions and states separately
            action_np = self.gaussian_normalizer.normalize_action(action_np)
            state_np = self.gaussian_normalizer.normalize_state(state_np)

        # ===== Pad to unified dimensions =====
        unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 288
        unified_state_dim = StateFeature.ALL_FEATURES[1]     # 296
        unified_state, unified_state_mask = pad_state_human(
            state_np,
            sample_dict["current_state_mask"],
            action_dim,
            state_dim,
            unified_state_dim
        )
        unified_action, unified_action_mask = pad_action(
            action_np,
            sample_dict["action_mask"],
            action_dim,
            unified_action_dim
        )
        (
            unified_action,
            unified_action_mask,
            unified_state,
            unified_state_mask,
        ) = self._apply_hand_feature_mode(
            unified_action, unified_action_mask, unified_state, unified_state_mask
        )

        sample_dict["action_list"] = unified_action
        sample_dict["action_mask"] = unified_action_mask
        sample_dict["current_state"] = unified_state
        sample_dict["current_state_mask"] = unified_state_mask
        return sample_dict

    def __getitem__(self, idx):
        data_id = int(self.sample_indices[idx])
        corr = self.index_frame_pair[data_id]
        episode_id = self.index_to_episode_id[corr[0]]
        frame_id = int(corr[1])
        sample = self.get_item_frame( # the second argument of the pair refers to the frame index
            episode_id, frame_id,
            action_past_window_size=self.action_past_window_size,
            action_future_window_size=self.action_future_window_size,
            image_past_window_size=self.image_past_window_size,
            image_future_window_size=self.image_future_window_size,
            rel_mode=self.rel_mode,  # 'step'
            load_images=self.load_images
        )
        episode_id = str(episode_id)
        sample["episode_id"] = episode_id
        sample["frame_id"] = frame_id
        sample["sample_id"] = canonical_sample_id(episode_id, frame_id)
        return sample

def pad_state_human(
    state: torch.Tensor,
    state_mask: torch.Tensor,
    action_dim: int, # 51+51=102 or 51+51+6=108
    state_dim: int, # 51+10+51+10=122 or 51+10+51+10+6=128
    unified_state_dim: int # 296
):
    """
    Expand state mask, mask invalid state dims, and pad current_state to a standard size.
    Supports 2 entities (left hand, right hand) or 3 entities (left hand, right hand, object).

    Args:
        current_state (Tensor): original state tensor, shape [state_dim]
        current_state_mask (Tensor): per-entity state mask, shape [2] or [3] (left, right, [obj])
        action_dim (int): original action dimension (102 or 108)
        state_dim (int): original state dimension (122 or 128)
        unified_state_dim (int): target padded state dimension (296)

    Returns:
        Tuple[Tensor, Tensor]:
            padded current_state [unified_state_dim],
            padded current_state_mask [unified_state_dim]
    """

    current_state = torch.tensor(state, dtype=torch.float32) # (122) or (128)
    current_state_mask = torch.tensor(state_mask, dtype=torch.bool) # (2,) or (3,)

    # Dimensions per entity
    left_hand_action_dim = 51  # 6 + 45
    right_hand_action_dim = 51
    object_action_dim = 6
    left_hand_state_dim = left_hand_action_dim + 10  # +10 for betas
    right_hand_state_dim = right_hand_action_dim + 10
    object_state_dim = object_action_dim  # No betas for object

    # Determine number of entities based on mask length
    num_entities = len(current_state_mask)
    has_object = num_entities == 3

    # Expand state mask from per-entity to per-dim
    if has_object:
        expanded_state_mask = torch.cat([
            current_state_mask[0].repeat(left_hand_state_dim),
            current_state_mask[1].repeat(right_hand_state_dim),
            current_state_mask[2].repeat(object_state_dim),
        ])
    else:
        expanded_state_mask = torch.cat([
            current_state_mask[0].repeat(left_hand_state_dim),
            current_state_mask[1].repeat(right_hand_state_dim),
        ])

    # Mask out invalid state dimensions
    current_state_masked = current_state * expanded_state_mask.to(current_state.dtype)

    # Initialize output tensors
    padded_state = torch.zeros(unified_state_dim, dtype=current_state.dtype) # 296
    padded_mask = torch.zeros(unified_state_dim, dtype=torch.bool)

    # Fill left hand state (skipping betas)
    padded_state[:left_hand_action_dim] = current_state_masked[:left_hand_action_dim].clone()
    padded_mask[:left_hand_action_dim] = expanded_state_mask[:left_hand_action_dim].clone()

    # Fill right hand state (skipping betas)
    padded_state[left_hand_action_dim:left_hand_action_dim+right_hand_action_dim] = \
        current_state_masked[left_hand_state_dim:left_hand_state_dim+right_hand_action_dim].clone()
    padded_mask[left_hand_action_dim:left_hand_action_dim+right_hand_action_dim] = \
        expanded_state_mask[left_hand_state_dim:left_hand_state_dim+right_hand_action_dim].clone()

    # Fill object state (only if has_object)
    if has_object:
        object_start_idx = left_hand_action_dim + right_hand_action_dim
        padded_state[object_start_idx:object_start_idx+object_action_dim] = \
            current_state_masked[left_hand_state_dim+right_hand_state_dim:left_hand_state_dim+right_hand_state_dim+object_action_dim].clone()
        padded_mask[object_start_idx:object_start_idx+object_action_dim] = \
            expanded_state_mask[left_hand_state_dim+right_hand_state_dim:left_hand_state_dim+right_hand_state_dim+object_action_dim].clone()

    return padded_state, padded_mask

def pad_action(
    actions: torch.Tensor = None,
    action_mask: torch.Tensor = None,
    action_dim: int = None,  # 102 (51+51) or 108 (51+51+6)
    unified_action_dim: int = None  # 288
):
    """
    Expand action mask per dimension, mask invalid actions, and pad actions to a unified size.
    Supports 2 entities (left hand, right hand) or 3 entities (left hand, right hand, object).

    Args:
        actions (Tensor or None): original actions tensor, shape [T, action_dim] or None.
        action_mask (Tensor): per-entity action mask, shape [T, 2] or [T, 3] (left, right, [obj]).
        action_dim (int): original action dimension (102 or 108).
        unified_action_dim (int): target padded actions dimension (288).

    Returns:
        Tuple[Optional[Tensor], Tensor]:
            padded actions [T, unified_action_dim] or None,
            padded action mask [T, unified_action_dim]
    """

    action_mask = torch.tensor(action_mask, dtype=torch.bool)

    # Dimensions per entity
    left_hand_dim = 51  # 6 + 45
    right_hand_dim = 51
    object_dim = 6

    # Determine number of entities based on mask shape
    num_entities = action_mask.shape[1]
    has_object = num_entities == 3

    # Expand mask from per-entity to per-dimension
    mask_left = action_mask[:, 0].unsqueeze(1).expand(-1, left_hand_dim)
    mask_right = action_mask[:, 1].unsqueeze(1).expand(-1, right_hand_dim)

    if has_object:
        mask_obj = action_mask[:, 2].unsqueeze(1).expand(-1, object_dim)
        expanded_action_mask = torch.cat((mask_left, mask_right, mask_obj), dim=1)
    else:
        expanded_action_mask = torch.cat((mask_left, mask_right), dim=1)

    # ---------------------------
    # Case 1: actions is None
    # ---------------------------
    if actions is None:
        padding_mask = torch.zeros(
            (action_mask.shape[0], unified_action_dim - action_dim),
            dtype=torch.bool
        )
        action_mask_padded = torch.cat((expanded_action_mask, padding_mask), dim=1)
        return None, action_mask_padded

    # ---------------------------
    # Case 2: actions exists
    # ---------------------------

    actions = torch.tensor(actions, dtype=torch.float32)
    # Mask invalid action dims
    actions_masked = actions * expanded_action_mask.to(actions.dtype)

    # Pad both actions and mask
    padding = torch.zeros(
        (actions.shape[0], unified_action_dim - action_dim),
        dtype=actions.dtype
    )

    actions_padded = torch.cat((actions_masked, padding), dim=1)
    action_mask_padded = torch.cat((expanded_action_mask, padding.bool()), dim=1)

    return actions_padded, action_mask_padded


class EpisodicEvalDataset(EpisodicDatasetCore):
    """
    Evaluation dataset that returns all future frames as ground truth action,
    instead of a fixed chunk.
    """
    # def __getitem__(self, idx):
    #     if self.training_idx is not None:
    #         data_id = self.training_idx[idx]
    #     else:
    #         data_id = idx
    #     corr = self.index_frame_pair[data_id]
    #     episode_id = self.index_to_episode_id[corr[0]]
    #     frame_id = int(corr[1])
        
    #     # Load episode to get total length T
    #     epi, _, _ = self._load_or_cache_episode(episode_id)
    #     T = len(epi['extrinsics'])
        
    #     # Calculate future window size to cover the rest of the episode
    #     # We want frames from [frame_id - past, T-1]
    #     # future size = (T-1) - frame_id
    #     # dynamic_future_window_size = max(0, T - 1 - frame_id)
    #     dynamic_future_window_size = self.action_future_window_size
        
    #     sample = self.get_item_frame(
    #         episode_id, frame_id,
    #         action_past_window_size=self.action_past_window_size,
    #         action_future_window_size=dynamic_future_window_size,
    #         image_past_window_size=self.image_past_window_size,
    #         image_future_window_size=self.image_future_window_size,
    #         rel_mode=self.rel_mode,
    #         load_images=self.load_images
    #     )
    #     return sample

    def get_item_frame(
            self, episode_id, frame_id,
            action_past_window_size=0,
            action_future_window_size=0,
            image_past_window_size=0,
            image_future_window_size=0,
            rel_mode: str = "step",
            load_images: bool = True,
        ):
        """
        Vectorised dataloader for evaluation - aligned with EpisodicDatasetCore.

        """
        # ------------------------------------------------------------------
        # 1. Load episode dict  +  extrinsics
        # ------------------------------------------------------------------
        epi, R_w2c, t_w2c = self._load_or_cache_episode(episode_id)
        T  = len(epi['extrinsics'])

        # ------------------------------------------------------------------
        # 2. Build frame-window indices
        # ------------------------------------------------------------------
        idx_win, oob  = self._window_indices(frame_id,
                                        action_past_window_size,
                                        action_future_window_size, 0, T-1)     # (W,)
        next_idx, next_oob = self._next_index_after_window(
            frame_id, action_future_window_size, 0, T - 1
        )
        W   = len(idx_win)

        # Check if object data exists in this episode
        has_object = self.use_obj and 'obj_rt' in epi

        obj_mesh_path = None
        if has_object and 'obj_mesh_path' in epi:
            obj_mesh_path = epi['obj_mesh_path']

        main_type = self._normalize_main_type(epi['anno_type'], epi.get('text'))
        sub_type = 'right' if main_type == 'left' else 'left'
        # ------------------------------------------------------------------
        # 3. Build instruction text
        # ------------------------------------------------------------------
        cot_text_selected = None
        if self.use_cot:
            instruction, idx_win_left, oob_left, idx_win_right, oob_right, \
            start_left, end_left, start_right, end_right, cot_text_selected = self._build_instruction(
                main_type = main_type,
                text_clip = epi['text'],
                text_rephrase = epi.get('text_rephrase'),
                idx_win = idx_win,
                oob = oob, # mask
                epi_len = T,
                frame_id = frame_id,
                action_past_window_size = action_past_window_size,
                action_future_window_size = action_future_window_size,
                cot = epi.get('cot', None) # List[Tuple(List[str], List[Tuple[int,int]])]] or None
            )
        else:
            instruction, idx_win_left, oob_left, idx_win_right, oob_right, \
            start_left, end_left, start_right, end_right, _ = self._build_instruction(
                main_type = main_type,
                text_clip = epi['text'],
                text_rephrase = epi.get('text_rephrase'),
                idx_win = idx_win,
                oob = oob, # mask
                epi_len = T,
                frame_id = frame_id,
                action_past_window_size = action_past_window_size,
                action_future_window_size = action_future_window_size,
            )

        next_idx_left, next_oob_left = self._next_index_after_window(
            frame_id, action_future_window_size, start_left, end_left
        )
        next_idx_right, next_oob_right = self._next_index_after_window(
            frame_id, action_future_window_size, start_right, end_right
        )

        # ------------------------------------------------------------------
        # 4. Vectorised actions  (left + right)
        # ------------------------------------------------------------------
        win_left  = self._prepare_side_window(
            epi['left'],  R_w2c, t_w2c, idx_win_left, frame_id, anchor_frame=True,
            oob=oob_left, start=start_left, end=end_left, upsample_factor=self.upsample_factor,
            next_idx=next_idx_left, next_oob=next_oob_left
        )
        win_right = self._prepare_side_window(
            epi['right'], R_w2c, t_w2c, idx_win_right, frame_id, anchor_frame=True,
            oob=oob_right, start=start_right, end=end_right, upsample_factor=self.upsample_factor,
            next_idx=next_idx_right, next_oob=next_oob_right
        )

        # Object window (use same window as main hand)
        if has_object:
            win_obj = self._prepare_object_window(
                epi['obj_rt'], R_w2c, t_w2c, idx_win, frame_id,
                oob=oob, start=0, end=T-1, upsample_factor=self.upsample_factor,
                next_idx=next_idx, next_oob=next_oob
            )
        else:
            # If no object data, create dummy window
            W = len(idx_win)
            win_obj = {
                'R_cam': np.zeros((W, 3, 3), dtype=np.float32),
                't_cam': np.zeros((W, 3), dtype=np.float32),
                'kept': np.zeros(W, dtype=bool),
                'R_cam_next': np.zeros((W, 3, 3), dtype=np.float32),
                't_cam_next': np.zeros((W, 3), dtype=np.float32),
                'kept_next': np.zeros(W, dtype=bool),
            }

        idx_center = action_past_window_size          # local index of t0 in window

        # rel_mode: "step"  or  "anchor" / action_type: "angle" or "keypoints"
        # step: relative to previous frame, anchor: relative to t0
        abs_L, rel_L, msk_L = self._make_action_window_vec(
            win_left,  anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
        )

        abs_R, rel_R, msk_R = self._make_action_window_vec(
            win_right, anchor_idx=idx_center, rel_mode=rel_mode, action_type=self.action_type
        )

        # Decide whether to include object based on has_object flag
        if has_object:
            abs_obj, rel_obj, msk_obj = self._make_object_action_window_vec(
                win_obj, anchor_idx=idx_center, rel_mode=rel_mode
            )
            action_abs = np.concatenate([abs_L, abs_R, abs_obj], axis=1)   # (W, 108) = 51+51+6
            action_rel = np.concatenate([rel_L, rel_R, rel_obj], axis=1)   # (W, 108)
            action_mask = np.stack([msk_L, msk_R, msk_obj], axis=1)        # (W, 3)
        else:
            action_abs = np.concatenate([abs_L, abs_R], axis=1)   # (W, 102) = 51+51
            action_rel = np.concatenate([rel_L, rel_R], axis=1)   # (W, 102)
            action_mask = np.stack([msk_L, msk_R], axis=1)        # (W, 2)

        cur_L = self._pack_state(win_left['R_cam'],
                    win_left['t_cam'],
                    win_left['pose_euler'] if self.action_type=='angle' else win_left['joints_manospace'].reshape(W, -1),
                    idx_center)

        cur_R = self._pack_state(win_right['R_cam'],
                            win_right['t_cam'],
                            win_right['pose_euler'] if self.action_type=='angle' else win_right['joints_manospace'].reshape(W, -1),
                            idx_center)


        betas_L = epi['left']['beta']
        betas_R = epi['right']['beta']

        # Decide whether to include object state based on has_object flag
        if has_object:
            # Object current state (6D: rotation + translation)
            cur_obj = np.concatenate([
                win_obj['t_cam'][idx_center],
                self._mat2euler(win_obj['R_cam'][idx_center][None, ...])[0]
            ])
            current_state = np.concatenate([cur_L, betas_L, cur_R, betas_R, cur_obj])  # 128 = 61+61+6
            current_state_mask = np.array([win_left['kept'][idx_center], win_right['kept'][idx_center], win_obj['kept'][idx_center]])
        else:
            current_state = np.concatenate([cur_L, betas_L, cur_R, betas_R])  # 122 = 61+61
            current_state_mask = np.array([win_left['kept'][idx_center], win_right['kept'][idx_center]])

        # ------------------------------------------------------------------
        # 5. RGB window
        # ------------------------------------------------------------------
        if load_images:
            image_list, image_mask = self._grab_window_images(
                episode_id, epi,
                frame_id,
                image_past_window_size,
                image_future_window_size
            )
            H = image_list[0].shape[0]
            W = image_list[0].shape[1]

            if not self.augmentation:
                 new_image_list = []
                 for img in image_list:
                     img_pil = Image.fromarray(img)
                     img_resized = resize_short_side_to_target(img_pil, target=self.target_image_width)
                     new_image_list.append(np.array(img_resized))
                 image_list = np.stack(new_image_list)
                 H = image_list[0].shape[0]
                 W = image_list[0].shape[1]
        else:
            image_list = None
            image_mask = None
            H, W = epi['intrinsics'][1,2]*2, epi['intrinsics'][0,2]*2
        # ------------------------------------------------------------------
        # 6. Calculate New_intrinsics
        # ------------------------------------------------------------------
        dataset_name = episode_id.split('_')[0]
        intrinsics = epi['intrinsics']

        if dataset_name == 'EgoExo4D':
            # For EgoExo4D, the fisheye camera images contain black borders after undistortion.
            # We remove these borders using a center crop. Specifically, the video frames are
            # first resized from 1408 to 448, and then center-cropped to 256.

            new_intrinsics = compute_new_intrinsics_crop(intrinsics, 1408, 256/448*1408, H)

        else:
            new_intrinsics = compute_new_intrinsics_resize(intrinsics, (H, W))

        # ------------------------------------------------------------------
        # 7. Do augmentation
        # ------------------------------------------------------------------
        if self.augmentation:
            try:
                # randomly sample aspect ratio for augmentation
                aspect_ratio = np.exp(random.uniform(np.log(1.0), np.log(2.0)))
                target_size = (int(self.target_image_width * aspect_ratio), self.target_image_width)  # (W, H)
                augment_params = {
                    'tgt_aspect': aspect_ratio,
                    'flip_augmentation': self.flip_augmentation,
                    'set_none_ratio': self.set_none_ratio,
                }

                uv_traj = self._get_2d_traj_cur_to_end(frame_id, epi, new_intrinsics, main_type, (H, W))
                image_list, new_intrinsics, (action_abs, action_rel, action_mask), \
                (current_state, current_state_mask), instruction = \
                    augmentation_func(
                        image = image_list,
                        intrinsics = new_intrinsics,
                        actions = (action_abs, action_rel, action_mask),
                        states = (current_state, current_state_mask),
                        captions = instruction,
                        uv_traj = uv_traj,
                        target_size = target_size,
                        augment_params = augment_params,
                        sub_type = sub_type,
                    )

            except Exception as e:
                print(f"Warning: Augmentation failed for episode {episode_id}, frame {frame_id}: {e}. Do center crop only")
                import traceback
                print(f"Warning: Augmentation failed for episode {episode_id}, frame {frame_id}")
                print(f"Exception: {type(e).__name__}: {e}")
                print(f"Traceback:\n{traceback.format_exc()}")
                image_list = center_crop_short_side(image_list[0])[None, ...]
                new_intrinsics[0][2] = 0.5 * image_list[0].shape[1]  # update the principal point
                new_intrinsics[1][2] = 0.5 * image_list[0].shape[0]  # update the principal point

            if random.random() < self.state_mask_prob:
                # Mask state based on has_object flag to maintain consistent dimensions
                if has_object:
                    current_state_mask = np.array([False, False, False])
                else:
                    current_state_mask = np.array([False, False])
                current_state[:] = 0.0

        fov = calculate_fov( 2 * new_intrinsics[1][2], 2 * new_intrinsics[0][2], new_intrinsics)

        if self.use_rel:
            action_list = action_rel
        else:
            # use abs action for hand pose only
            # When has_object: shape is (W, 108) = 51(L) + 51(R) + 6(obj)
            # When no object:  shape is (W, 102) = 51(L) + 51(R)
            if has_object:
                rel_L = action_rel[:, :51]
                rel_R = action_rel[:, 51:102]
                abs_L = action_abs[:, :51]
                abs_R = action_abs[:, 51:102]
                rel_obj = action_rel[:, 102:]
                abs_obj = action_abs[:, 102:]
                if self.use_obj_rel:
                    action_list = np.concatenate([rel_L[:, :6], abs_L[:, 6:], rel_R[:, :6], abs_R[:, 6:], rel_obj], axis=1)
                else:
                    action_list = np.concatenate([rel_L[:, :6], abs_L[:, 6:], rel_R[:, :6], abs_R[:, 6:], abs_obj], axis=1)
            else:
                rel_L = action_rel[:, :action_rel.shape[1]//2]
                rel_R = action_rel[:, action_rel.shape[1]//2:]
                abs_L = action_abs[:, :action_abs.shape[1]//2]
                abs_R = action_abs[:, action_abs.shape[1]//2:]
                action_list = np.concatenate([rel_L[:, :6], abs_L[:, 6:], rel_R[:, :6], abs_R[:, 6:]], axis=1)


        # ------------------------------------------------------------------
        # 8. Return to caller
        # ------------------------------------------------------------------

        result_dict = dict(
            instruction             = instruction,
            action_list             = action_list,          # (W,2*51) or (W,2*51+6) float32
            action_mask             = action_mask,          # (W,2) bool or (W,3) bool
            current_state           = current_state,        # (2*61,) or (2*61+6,) float32
            current_state_mask      = current_state_mask,   # (2,) bool or (3,) bool
            fov                     = fov,                  # (2,) float32
            intrinsics              = new_intrinsics,       # (3,3) float32
        )
        # Only add obj_mesh_path if it's not None (to avoid collate issues)
        if obj_mesh_path is not None:
            result_dict['obj_mesh_path'] = obj_mesh_path
        if cot_text_selected is not None:
            result_dict['cot'] = cot_text_selected

        if image_list is not None:
            result_dict['image_list'] = image_list          # (W,H,W,3) uint8
        if image_mask is not None:
            result_dict['image_mask'] = image_mask          # (W,) bool

        return result_dict
