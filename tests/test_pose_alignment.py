import unittest

import numpy as np

from empire.models.depth_anything_3.utils.geometry import affine_inverse_np
from empire.models.depth_anything_3.utils.pose_align import align_poses_umeyama


class PoseAlignmentTest(unittest.TestCase):
    def test_umeyama_alignment_executes_evo_path_and_recovers_trajectory(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.5, -0.5, 2.0],
            ],
            dtype=np.float64,
        )
        angle = 0.37
        sim_rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        sim_scale = 1.7
        sim_translation = np.array([0.4, -1.2, 2.5], dtype=np.float64)

        reference_poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(points), axis=0)
        reference_poses[:, :3, 3] = points
        estimated_poses = reference_poses.copy()
        estimated_poses[:, :3, :3] = sim_rotation @ reference_poses[:, :3, :3]
        estimated_poses[:, :3, 3] = (
            sim_scale * (sim_rotation @ points.T).T + sim_translation
        )

        reference_extrinsics = affine_inverse_np(reference_poses)[:, :3, :]
        estimated_extrinsics = affine_inverse_np(estimated_poses)[:, :3, :]
        _, _, _, aligned_extrinsics = align_poses_umeyama(
            reference_extrinsics,
            estimated_extrinsics,
            return_aligned=True,
        )

        aligned_poses = affine_inverse_np(aligned_extrinsics)
        np.testing.assert_allclose(aligned_poses, reference_poses, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
