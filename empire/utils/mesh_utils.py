"""
Utility functions for loading meshes and computing distances between hands and objects.
"""
import os
import torch
import numpy as np
from typing import Optional, Tuple, List
import trimesh
from scipy.spatial.transform import Rotation as R


def load_mesh_and_sample_points(mesh_path: str, num_points: int = 1000, device: str = 'cuda') -> Optional[torch.Tensor]:
    """
    Load a mesh file and sample points from its surface.

    Args:
        mesh_path: Path to the mesh file (supports .obj, .ply, .stl, etc.)
        num_points: Number of points to sample from the mesh surface
        device: Torch device to place the tensor on

    Returns:
        Tensor of shape (num_points, 3) containing sampled points, or None if loading fails
    """
    if mesh_path is None or mesh_path == 'empty path' or not os.path.exists(mesh_path):
        return None

    try:
        # Load mesh using trimesh
        mesh = trimesh.load(mesh_path)

        # Sample points from mesh surface
        # Trimesh samples uniformly from the surface area
        # points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        points = mesh.vertices / 1000.0  # Convert from mm to meters if needed

        # Convert to torch tensor
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

        return points_tensor
    except Exception as e:
        print(f"Warning: Failed to load or sample mesh from {mesh_path}: {e}")
        return None


def compute_hand_object_distance(
    hand_keypoints: torch.Tensor,
    object_points: torch.Tensor,
    mode: str = 'min'
) -> torch.Tensor:
    """
    Compute the minimum distance between hand keypoints and object point cloud.

    Args:
        hand_keypoints: Hand keypoints of shape (B, T, num_hand_joints, 3)
        object_points: Object point cloud of shape (num_points, 3)
        mode: 'min' for minimum distance, 'mean' for mean distance

    Returns:
        Distance tensor of shape (B, T)
    """
    B, T, num_joints, _ = hand_keypoints.shape

    # Reshape hand keypoints for pairwise distance computation
    # (B, T, num_joints, 3) -> (B*T, num_joints, 3)
    hand_reshaped = hand_keypoints.reshape(B * T, num_joints, 3)

    # Expand dimensions for broadcasting
    # hand: (B*T, num_joints, 1, 3)
    # object: (1, 1, num_points, 3)
    hand_expanded = hand_reshaped.unsqueeze(2)  # (B*T, num_joints, 1, 3)
    object_expanded = object_points.unsqueeze(0).unsqueeze(0)  # (1, 1, num_points, 3)

    # Compute pairwise distances
    # (B*T, num_joints, num_points)
    distances = torch.norm(hand_expanded - object_expanded, dim=-1)

    # Compute minimum distance across all hand joints and object points
    # (B*T,)
    if mode == 'min':
        min_distances = distances.min(dim=2)[0]  # min over object points: (B*T, num_joints)
        min_distances = min_distances.min(dim=1)[0]  # min over hand joints: (B*T,)
    elif mode == 'mean':
        mean_distances = distances.mean(dim=2)  # mean over object points: (B*T, num_joints)
        min_distances = mean_distances.min(dim=1)[0]  # min over hand joints: (B*T,)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Reshape back to (B, T)
    min_distances = min_distances.reshape(B, T)

    return min_distances


def actions_to_hand_keypoints(
    actions: torch.Tensor,
    action_type: str = 'angle',
    mano_model=None,
    hand_side: str = 'right'
) -> torch.Tensor:
    """
    Convert action representations to hand keypoints (3D joint positions).

    Args:
        actions: Action tensor of shape (B, T, action_dim)
        action_type: Type of action representation ('angle' or 'keypoints')
        mano_model: Optional MANO model for converting angles to keypoints
        hand_side: 'left' or 'right' hand

    Returns:
        Hand keypoints tensor of shape (B, T, num_joints, 3)
    """
    B, T, action_dim = actions.shape

    if action_type == 'keypoints':
        # If already keypoints, just reshape
        # Assuming keypoints are flattened in the action vector
        num_joints = 21  # MANO has 21 joints
        keypoints = actions[:, :, :num_joints*3].reshape(B, T, num_joints, 3)
        return keypoints

    elif action_type == 'angle':
        # Need to convert angle representation to keypoints using MANO
        if mano_model is None:
            raise ValueError("MANO model is required to convert angles to keypoints")

        # Extract hand parameters from actions
        # For right hand: translation (51:54), rotation (54:57), joints (57:102)
        # For left hand: translation (0:3), rotation (3:6), joints (6:51)
        if hand_side == 'right':
            trans_idx = (51, 54)
            rot_idx = (54, 57)
            joints_idx = (57, 102)
        elif hand_side == 'left':
            trans_idx = (0, 3)
            rot_idx = (3, 6)
            joints_idx = (6, 51)
        else:
            raise ValueError(f"Unknown hand side: {hand_side}")

        # Extract parameters
        # Assuming actions are normalized, need to denormalize first
        # This is a simplified version - you may need to adjust based on your normalization
        hand_trans = actions[:, :, trans_idx[0]:trans_idx[1]]  # (B, T, 3)
        hand_rot = actions[:, :, rot_idx[0]:rot_idx[1]]  # (B, T, 3)
        hand_joints = actions[:, :, joints_idx[0]:joints_idx[1]]  # (B, T, 45)

        # Convert to numpy for MANO (MANO expects numpy arrays or specific tensor format)
        # This is a placeholder - actual implementation depends on your MANO setup
        # You may need to convert 6D rotation to rotation matrix, etc.

        # For now, return a placeholder
        # TODO: Implement actual MANO forward pass
        print("Warning: MANO conversion not fully implemented, returning keypoints from action if available")
        num_joints = 21
        keypoints = torch.zeros(B, T, num_joints, 3, device=actions.device)
        return keypoints

    else:
        raise ValueError(f"Unknown action type: {action_type}")


def convert_trajectory_to_keypoints(
    traj: np.ndarray,
    mano_model,
    is_left: bool = False
) -> np.ndarray:
    """
    Convert hand trajectory (MANO parameters) to 3D keypoints using MANO model.

    Args:
        traj: Hand trajectory of shape (T, 51) containing:
            - translation (3)
            - global rotation euler (3)
            - hand pose euler (45)
        mano_model: MANO model instance
        is_left: Whether this is left hand

    Returns:
        keypoints: 3D keypoints of shape (T, 21, 3)
    """
    import torch

    T = traj.shape[0]
    keypoints_list = []

    for t in range(T):
        # Extract MANO parameters
        transl = traj[t, 0:3]  # [3]
        global_orient_euler = traj[t, 3:6]  # [3]
        hand_pose_euler = traj[t, 6:51].reshape(15, 3)  # [15, 3]

        # Convert Euler to rotation matrices
        global_orient_mat = R.from_euler('xyz', global_orient_euler).as_matrix().reshape(1, 3, 3)
        hand_pose_mat = R.from_euler('xyz', hand_pose_euler).as_matrix().reshape(1, 15, 3, 3)

        # Convert to torch tensors for MANO
        global_orient_mat = torch.tensor(global_orient_mat, dtype=torch.float32, device=mano_model.transl.device)
        hand_pose_mat = torch.tensor(hand_pose_mat, dtype=torch.float32, device=mano_model.transl.device)
        transl = torch.tensor(transl, dtype=torch.float32, device=mano_model.transl.device).reshape(1, 3)

        # Run MANO forward pass
        mano_output = mano_model(
            global_orient=global_orient_mat,
            hand_pose=hand_pose_mat,
            transl=transl,
            pose2rot=False
        )

        # Extract keypoints (joints)
        keypoints_tensor = mano_output.joints[0]  # [21, 3]

        # Convert to numpy, handling both tensors with and without grad
        if isinstance(keypoints_tensor, torch.Tensor):
            if keypoints_tensor.requires_grad:
                keypoints = keypoints_tensor.detach().cpu().numpy()
            else:
                keypoints = keypoints_tensor.cpu().numpy()
        else:
            keypoints = keypoints_tensor

        keypoints_list.append(keypoints)

    return np.stack(keypoints_list)  # [T, 21, 3]


def transform_object_trajectory(
    traj: np.ndarray,
    mesh_path: str,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Apply object trajectory transformations to mesh vertices.

    Args:
        traj: Object trajectory of shape (T, 6) containing:
            - translation (3)
            - rotation euler (3)
        mesh_path: Path to the object mesh file
        device: Torch device

    Returns:
        transformed_vertices: Transformed mesh vertices of shape (T, V, 3)
                              where V is the number of vertices
    """
    # Load base mesh
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    try:
        mesh = trimesh.load(mesh_path)
        base_vertices = mesh.vertices  # [V, 3]
        # Convert from mm to meters if needed (as in inference_human.py)
        base_vertices = base_vertices / 1000.0  # [V, 3]
    except Exception as e:
        raise RuntimeError(f"Failed to load mesh from {mesh_path}: {e}")

    T = traj.shape[0]
    V = base_vertices.shape[0]
    transformed_vertices = np.zeros((T, V, 3), dtype=np.float32)

    for t in range(T):
        # Extract translation and rotation
        transl = traj[t, 0:3]  # [3]
        rot_euler = traj[t, 3:6]  # [3]

        # Convert Euler to rotation matrix
        rot_mat = R.from_euler('xyz', rot_euler).as_matrix()  # [3, 3]

        # Apply rotation and translation to vertices
        # vertices_transformed = rot_mat @ vertices.T + transl
        # Which is: vertices @ rot_mat.T + transl
        vertices_rotated = base_vertices @ rot_mat.T  # [V, 3]
        vertices_transformed = vertices_rotated + transl  # [V, 3]

        transformed_vertices[t] = vertices_transformed

    return torch.tensor(transformed_vertices, dtype=torch.float32, device=device)


def obj_traj_abs(state, abs_action):
    """
    Reconstruct the object trajectory using absolute transformations.

    This function takes an initial object state and a sequence of absolute
    actions (transformations) to compute the complete object trajectory.
    Unlike relative trajectory reconstruction, each action contains the
    absolute transformation for that timestep, not the delta from the previous state.

    Args:
        state (np.ndarray): Current object state [6] containing:
            - Translation [3]: Object position in 3D space
            - Rotation [3]: Object rotation as Euler angles (xyz format)
        abs_action (np.ndarray): Absolute action sequence [T, 6] containing:
            - Absolute translation [3] per timestep
            - Absolute rotation [3] as Euler angles per timestep

    Returns:
        np.ndarray: Complete object trajectory [T+1, 6] where:
            - First row is the initial state
            - Subsequent rows are the absolute transformations at each timestep
            - Each row contains [translation(3), rotation_euler(3)]
    """
    traj_list = []

    # Add initial state (step 0)
    traj_list.append(state.copy())

    # Process each timestep - actions are already absolute, no accumulation needed
    for t in range(abs_action.shape[0]):
        # Extract absolute translation and rotation from action
        abs_t = abs_action[t, :3]  # [3] - absolute translation
        abs_R_euler = abs_action[t, 3:6]  # [3] - absolute rotation as euler angles

        # Simply concatenate - no accumulation since these are absolute transformations
        action = np.concatenate([abs_t, abs_R_euler])  # [6]
        traj_list.append(action)

    return np.stack(traj_list)  # Shape: [T+1, 6]


def compute_hand_object_mse_loss_from_actions(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    current_state: torch.Tensor,
    obj_mesh_paths: List[str],
    mano_model_right,
    mano_model_left,
    action_masks: torch.Tensor,
    action_type: str = 'angle',
    use_rel: bool = False,
    rel_mode: str = 'step',
    use_obj_rel: bool = False,
    num_sample_points: int = 1000,
    device: str = 'cuda'
) -> Optional[torch.Tensor]:
    """
    Compute MSE loss between predicted and ground truth hand-object distances.
    This version properly handles relative/absolute actions and uses MANO for conversion.
    Supports both left and right hands, and respects action_masks.

    Args:
        pred_actions: Predicted actions (relative or absolute) of shape (B*T, action_dim)
        gt_actions: Ground truth actions (relative) of shape (B*T, action_dim)
        current_state: Current state for each sample (B*T, state_dim)
        obj_mesh_paths: List of object mesh paths for each sample
        mano_model_right: Right hand MANO model
        mano_model_left: Left hand MANO model
        action_masks: Action mask of shape (B*T, action_dim), 1=compute loss, 0=ignore
        action_type: 'angle' or 'keypoints'
        use_rel: Whether actions are relative (True) or absolute (False)
        rel_mode: 'step' or 'anchor' mode for relative accumulation
        use_obj_rel: Whether object actions are relative (True) or absolute (False)
        num_sample_points: Number of points to sample from object mesh
        device: Torch device

    Returns:
        MSE loss scalar, or None if no valid meshes
    """
    from empire.utils.data_utils import recon_traj, recon_traj_object

    batch_size = pred_actions.shape[0]

    # Process each sample in batch
    losses = []
    for i in range(batch_size):
        mesh_path = obj_mesh_paths[i] if i < len(obj_mesh_paths) else None

        if mesh_path is None or mesh_path == 'empty path' or not os.path.exists(mesh_path):
            continue

        # Load object mesh and sample points
        object_points = load_mesh_and_sample_points(mesh_path, num_sample_points, device)
        if object_points is None:
            continue

        # Check action masks to determine which hands to compute loss for
        # Action format: [left(51), right(51), obj(6)]
        mask_left = action_masks[i, 0:51].any().item()  # True if any left hand action is masked
        mask_right = action_masks[i, 51:102].any().item()  # True if any right hand action is masked
        mask_obj = action_masks[i, 102:108].any().item()  # True if object action is masked

        # Skip if both hands are masked out (nothing to compute)
        if not mask_left and not mask_right:
            continue

        # State format with object: [left(51), left_beta(10), right(51), right_beta(10), obj(6)]

        if action_type == 'angle':
            sample_losses = []

            # ==================== Process Left Hand ====================
            if mask_left and mano_model_left is not None:
                # Extract left hand actions [51]
                pred_action_left = pred_actions[i, 0:51].cpu().numpy()  # [51]
                gt_action_left = gt_actions[i, 0:51].cpu().numpy()  # [51]

                # Extract left hand state [51]
                state_left = current_state[i, 0:51].cpu().numpy()  # [51]

                # Reshape for recon_traj: need [T, 51]
                pred_action_left = pred_action_left.reshape(1, -1)  # [1, 51]
                gt_action_left = gt_action_left.reshape(1, -1)  # [1, 51]

                # Reconstruct left hand trajectory
                pred_traj_left = recon_traj(state_left, pred_action_left, abs_joint=True, rel_mode=rel_mode)  # [T+1, 51]
                gt_traj_left = recon_traj(state_left, gt_action_left, abs_joint=True, rel_mode=rel_mode)  # [T+1, 51]

                # Convert left hand trajectory to keypoints
                pred_keypoints_left = convert_trajectory_to_keypoints(pred_traj_left, mano_model_left, is_left=True)  # [T+1, 21, 3]
                gt_keypoints_left = convert_trajectory_to_keypoints(gt_traj_left, mano_model_left, is_left=True)  # [T+1, 21, 3]

                pred_keypoints_left = torch.tensor(pred_keypoints_left, dtype=torch.float32, device=device)
                gt_keypoints_left = torch.tensor(gt_keypoints_left, dtype=torch.float32, device=device)

                # Keep for later processing
                pred_keypoints_left_processed = pred_keypoints_left
                gt_keypoints_left_processed = gt_keypoints_left
            else:
                pred_keypoints_left_processed = None
                gt_keypoints_left_processed = None

            # ==================== Process Right Hand ====================
            if mask_right and mano_model_right is not None:
                # Extract right hand actions [51]
                pred_action_right = pred_actions[i, 51:102].cpu().numpy()  # [51]
                gt_action_right = gt_actions[i, 51:102].cpu().numpy()  # [51]

                # Extract right hand state [51]
                # Skip left_beta (10) at indices [51:61]
                state_right = current_state[i, 61:112].cpu().numpy()  # [51]

                # Reshape for recon_traj: need [T, 51]
                pred_action_right = pred_action_right.reshape(1, -1)  # [1, 51]
                gt_action_right = gt_action_right.reshape(1, -1)  # [1, 51]

                # Reconstruct right hand trajectory
                pred_traj_right = recon_traj(state_right, pred_action_right, abs_joint=True, rel_mode=rel_mode)  # [T+1, 51]
                gt_traj_right = recon_traj(state_right, gt_action_right, abs_joint=True, rel_mode=rel_mode)  # [T+1, 51]

                # Convert right hand trajectory to keypoints
                pred_keypoints_right = convert_trajectory_to_keypoints(pred_traj_right, mano_model_right, is_left=False)  # [T+1, 21, 3]
                gt_keypoints_right = convert_trajectory_to_keypoints(gt_traj_right, mano_model_right, is_left=False)  # [T+1, 21, 3]

                pred_keypoints_right = torch.tensor(pred_keypoints_right, dtype=torch.float32, device=device)
                gt_keypoints_right = torch.tensor(gt_keypoints_right, dtype=torch.float32, device=device)

                # Keep for later processing
                pred_keypoints_right_processed = pred_keypoints_right
                gt_keypoints_right_processed = gt_keypoints_right
            else:
                pred_keypoints_right_processed = None
                gt_keypoints_right_processed = None

            # Skip if no valid hands
            if pred_keypoints_left_processed is None and pred_keypoints_right_processed is None:
                continue

            # ==================== Process Object ====================
            # Only process object if at least one hand is active
            # Extract object actions [6]
            pred_action_obj = pred_actions[i, 102:108].cpu().numpy()  # [6]
            gt_action_obj = gt_actions[i, 102:108].cpu().numpy()  # [6]

            # Extract object state [6]
            # Skip right_beta (10) at indices [112:122]
            state_obj = current_state[i, 122:128].cpu().numpy()  # [6]

            # Reshape object actions: need [T, 6]
            pred_action_obj = pred_action_obj.reshape(1, -1)  # [1, 6]
            gt_action_obj = gt_action_obj.reshape(1, -1)  # [1, 6]

            # Reconstruct object trajectories based on use_obj_rel flag
            if use_obj_rel:
                # Relative mode: use recon_traj_object
                pred_traj_obj = recon_traj_object(state_obj, pred_action_obj, rel_mode=rel_mode)  # [T+1, 6]
                gt_traj_obj = recon_traj_object(state_obj, gt_action_obj, rel_mode=rel_mode)  # [T+1, 6]
            else:
                # Absolute mode: use obj_traj_abs
                pred_traj_obj = obj_traj_abs(state_obj, pred_action_obj)  # [T+1, 6]
                gt_traj_obj = obj_traj_abs(state_obj, gt_action_obj)  # [T+1, 6]

            # Transform object trajectory to 3D mesh positions
            pred_obj_points = transform_object_trajectory(pred_traj_obj, mesh_path, device)  # [T+1, V, 3]
            gt_obj_points = transform_object_trajectory(gt_traj_obj, mesh_path, device)  # [T+1, V, 3]

            # ==================== Compute Distances ====================
            # Compute distances for each hand that is available and not masked

            # Left hand distance
            if pred_keypoints_left_processed is not None and gt_keypoints_left_processed is not None:
                pred_keypoints_left_b = pred_keypoints_left_processed.unsqueeze(0)  # [1, T+1, 21, 3]
                gt_keypoints_left_b = gt_keypoints_left_processed.unsqueeze(0)  # [1, T+1, 21, 3]
                pred_obj_points_b = pred_obj_points.unsqueeze(0)  # [1, T+1, V, 3]
                gt_obj_points_b = gt_obj_points.unsqueeze(0)  # [1, T+1, V, 3]

                pred_dist_left = compute_hand_object_distance(
                    pred_keypoints_left_b[:, -1:],
                    pred_obj_points_b[:, -1].reshape(-1, 3),
                    mode='min'
                )
                gt_dist_left = compute_hand_object_distance(
                    gt_keypoints_left_b[:, -1:],
                    gt_obj_points_b[:, -1].reshape(-1, 3),
                    mode='min'
                )
                loss_left = torch.mean((pred_dist_left - gt_dist_left) ** 2)
                sample_losses.append(loss_left)

            # Right hand distance
            if pred_keypoints_right_processed is not None and gt_keypoints_right_processed is not None:
                pred_keypoints_right_b = pred_keypoints_right_processed.unsqueeze(0)  # [1, T+1, 21, 3]
                gt_keypoints_right_b = gt_keypoints_right_processed.unsqueeze(0)  # [1, T+1, 21, 3]
                pred_obj_points_b = pred_obj_points.unsqueeze(0)  # [1, T+1, V, 3]
                gt_obj_points_b = gt_obj_points.unsqueeze(0)  # [1, T+1, V, 3]

                pred_dist_right = compute_hand_object_distance(
                    pred_keypoints_right_b[:, -1:],
                    pred_obj_points_b[:, -1].reshape(-1, 3),
                    mode='min'
                )
                gt_dist_right = compute_hand_object_distance(
                    gt_keypoints_right_b[:, -1:],
                    gt_obj_points_b[:, -1].reshape(-1, 3),
                    mode='min'
                )
                loss_right = torch.mean((pred_dist_right - gt_dist_right) ** 2)
                print(f"Sample {i}: Right hand MSE loss = {loss_right.item():.6f}")
                sample_losses.append(loss_right)

            # Average over available hands for this sample
            if len(sample_losses) > 0:
                sample_loss = torch.stack(sample_losses).mean()
                losses.append(sample_loss)

    if len(losses) == 0:
        return None

    # Average over all valid samples
    return torch.stack(losses).mean()
