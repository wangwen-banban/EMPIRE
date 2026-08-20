import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compute_egodex_metrics import side_errors


class SideErrorsTest(unittest.TestCase):
    def test_wrist_and_finger_relative_errors_are_separated(self):
        gt_joints = np.zeros((2, 21, 3), dtype=np.float32)
        pred_joints = gt_joints.copy()

        # A global translation affects global and wrist MPJPE, but cancels in
        # wrist-relative finger coordinates.
        pred_joints[0] += np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # A finger-only offset affects global and finger-relative MPJPE, but not
        # wrist MPJPE.
        pred_joints[1, 1:] += np.array([0.0, 2.0, 0.0], dtype=np.float32)

        pred = {"left_joints": pred_joints, "right_joints": pred_joints.copy()}
        gt = {"left_joints": gt_joints, "right_joints": gt_joints.copy()}
        mask = np.array([[True, False], [True, False]])

        metrics = side_errors(pred, gt, mask)

        self.assertAlmostEqual(metrics["left_wrist_mpjpe"], 0.5)
        self.assertAlmostEqual(metrics["left_finger_relative_mpjpe"], 1.0)
        self.assertAlmostEqual(
            metrics["left_mpjpe"],
            (1.0 + 40.0 / 21.0) / 2.0,
            places=6,
        )
        self.assertAlmostEqual(metrics["wrist_mpjpe"], 0.5)
        self.assertAlmostEqual(metrics["finger_relative_mpjpe"], 1.0)


if __name__ == "__main__":
    unittest.main()
