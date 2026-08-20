import unittest

import torch

from empire.models.action_model.diffusion_policy import DiffusionPolicy


class DiffusionPolicyLossWeightsTest(unittest.TestCase):
    def test_zero_weight_finger_components_do_not_contribute_gradients(self):
        policy = DiffusionPolicy(
            token_size=8,
            model_type="DiT-T",
            in_channels=288,
            future_action_window_size=1,
            loss_component_weights={
                "left_hand_joints": 0.0,
                "right_hand_joints": 0.0,
            },
        )
        pred = torch.zeros(1, 2, 288, requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.zeros_like(pred, dtype=torch.bool)

        mask[:, :, :102] = True
        target[:, :, 0:6] = 1.0
        target[:, :, 6:51] = float(10)
        target[:, :, 51:57] = 1.0
        target[:, :, 57:102] = float(10)

        losses = policy._masked_component_losses(pred, target, mask)
        self.assertAlmostEqual(losses["loss"].item(), 1.0)
        self.assertAlmostEqual(losses["left_hand_6d"].item(), 1.0)
        self.assertAlmostEqual(losses["right_hand_6d"].item(), 1.0)
        self.assertAlmostEqual(losses["left_hand_joints"].item(), 0.0)
        self.assertAlmostEqual(losses["right_hand_joints"].item(), 0.0)

        losses["loss"].backward()
        self.assertGreater(pred.grad[:, :, 0:6].abs().sum().item(), 0.0)
        self.assertEqual(pred.grad[:, :, 6:51].abs().sum().item(), 0.0)
        self.assertGreater(pred.grad[:, :, 51:57].abs().sum().item(), 0.0)
        self.assertEqual(pred.grad[:, :, 57:102].abs().sum().item(), 0.0)

    def test_unknown_component_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown loss component"):
            DiffusionPolicy(
                token_size=8,
                model_type="DiT-T",
                in_channels=288,
                future_action_window_size=1,
                loss_component_weights={"left_fingers": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
