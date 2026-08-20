import json
import tempfile
import unittest
from pathlib import Path

from empire.datasets.dataset import FrameDataset
from empire.datasets.human_dataset import EpisodicDatasetCore

import numpy as np


def make_dataset(plan_source, sidecar=None):
    dataset = FrameDataset.__new__(FrameDataset)
    dataset.cot_source = plan_source
    dataset.predicted_plan_sidecar_path = str(sidecar) if sidecar is not None else None
    dataset._predicted_plan_map = None
    return dataset


class PredictedPlanSourceTest(unittest.TestCase):
    def test_runtime_core_adds_canonical_sample_id(self):
        core = EpisodicDatasetCore.__new__(EpisodicDatasetCore)
        core.sample_indices = np.array([0], dtype=np.int64)
        core.index_frame_pair = np.array([[0, 4]], dtype=np.int64)
        core.index_to_episode_id = np.array(["ep_1"], dtype=object)
        core.action_past_window_size = 0
        core.action_future_window_size = 0
        core.image_past_window_size = 0
        core.image_future_window_size = 0
        core.rel_mode = "step"
        core.load_images = False
        core.get_item_frame = lambda *args, **kwargs: {"plan": "reach"}

        sample = EpisodicDatasetCore.__getitem__(core, 0)
        self.assertEqual(sample["episode_id"], "ep_1")
        self.assertEqual(sample["frame_id"], 4)
        self.assertEqual(sample["sample_id"], "ep_1:4")

    def test_reference_source_preserves_plan(self):
        dataset = make_dataset("gt")
        sample = {"plan": "reach and grasp"}
        self.assertEqual(dataset._select_cot(sample), "reach and grasp")
        self.assertEqual(sample, {"plan": "reach and grasp"})

    def test_predicted_source_uses_sample_id(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "predicted.json"
            sidecar.write_text(
                json.dumps({"schema_version": "1.0", "plans": {"ep_1:4": {"plan": "predicted motion"}}}),
                encoding="utf-8",
            )
            dataset = make_dataset("predicted", sidecar)
            sample = {"plan": "reference motion", "episode_id": "ep_1", "frame_id": 4}
            self.assertEqual(dataset._select_cot(sample), "predicted motion")
            self.assertEqual(sample["plan"], "reference motion")
            self.assertEqual(sample["predicted_plan"], "predicted motion")

    def test_predicted_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "predicted.json"
            sidecar.write_text(
                json.dumps({"schema_version": "1.0", "plans": {"other:0": "another plan"}}),
                encoding="utf-8",
            )
            dataset = make_dataset("predicted", sidecar)
            with self.assertRaises(KeyError):
                dataset._select_cot({"plan": "reference", "sample_id": "missing:1"})

    def test_reference_plan_text_is_not_a_sidecar_lookup_key(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "predicted.json"
            sidecar.write_text(
                json.dumps({"schema_version": "1.0", "plans": {"other:0": "reference motion"}}),
                encoding="utf-8",
            )
            dataset = make_dataset("predicted", sidecar)
            with self.assertRaises(KeyError):
                dataset._select_cot(
                    {"plan": "reference motion", "episode_id": "missing", "frame_id": 1}
                )

    def test_noncanonical_runtime_sample_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "predicted.json"
            sidecar.write_text(
                json.dumps({"schema_version": "1.0", "plans": {"ep:4": "move"}}),
                encoding="utf-8",
            )
            dataset = make_dataset("predicted", sidecar)
            with self.assertRaisesRegex(ValueError, "canonical frame_id"):
                dataset._select_cot({"plan": "reference", "sample_id": "ep:04"})

    def test_missing_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_dataset("predicted", Path(directory) / "missing.json")
            with self.assertRaises(FileNotFoundError):
                dataset._select_cot({"plan": "reference", "sample_id": "ep:0"})


if __name__ == "__main__":
    unittest.main()
