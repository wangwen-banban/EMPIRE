"""Optional MANO adapter built on the separately installed ``smplx`` package."""

from __future__ import annotations

from pathlib import Path
from typing import Any


# OpenPose's public 21-joint order over the 16 MANO joints plus five tips.
_OPENPOSE_JOINT_MAP = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
_FINGERTIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def create_mano_layer(
    model_path: str | Path,
    *,
    is_right: bool,
    device: str | Any | None = None,
):
    """Create an SMPL-X MANO layer that returns joints in OpenPose order.

    MANO assets are never downloaded by this package. ``model_path`` must point
    to assets obtained separately under the publisher's terms.
    """
    if not model_path:
        raise ValueError("A MANO model path is required")
    expanded_path = Path(model_path).expanduser()
    if not expanded_path.exists():
        raise FileNotFoundError(f"MANO model path does not exist: {expanded_path}")

    import torch
    from smplx import MANO as SMPLXMANO
    from smplx.vertex_ids import vertex_ids

    class MANOOpenPoseJoints(SMPLXMANO):
        def __init__(self):
            super().__init__(
                model_path=str(expanded_path),
                is_rhand=is_right,
                use_pca=False,
                flat_hand_mean=False,
            )
            tip_ids = [vertex_ids["mano"][name] for name in _FINGERTIP_NAMES]
            self.register_buffer("public_tip_ids", torch.tensor(tip_ids, dtype=torch.long))
            self.register_buffer(
                "public_joint_map",
                torch.tensor(_OPENPOSE_JOINT_MAP, dtype=torch.long),
            )

        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            tips = torch.index_select(output.vertices, 1, self.public_tip_ids)
            output.joints = torch.cat((output.joints, tips), dim=1)[:, self.public_joint_map]
            return output

    layer = MANOOpenPoseJoints()
    return layer.to(device) if device is not None else layer
