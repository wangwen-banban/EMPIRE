import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from empire.datasets.dataset import FrameDataset
from empire.datasets.schema import (
    canonical_sample_id,
    load_dataset_index,
    load_episode_metadata,
    load_predicted_plan_sidecar,
    resolve_dataset_layout,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from egodex_eval_utils import load_episode as eval_load_episode
from summarize_egodex_all_tasks import (
    test_episode_tasks as summarize_test_episode_tasks,
    training_task_counts,
)


class DataSchemaTest(unittest.TestCase):
    @staticmethod
    def _write_layout(root: Path, dataset_name: str, *, legacy: bool) -> tuple[Path, str]:
        episode_id = "fixture_episode"
        if legacy:
            dataset_dir = root / "Annotation" / dataset_name
            annotations = dataset_dir / "processed_episodes_vitra_format"
            index_file = dataset_dir / "episode_frame_index.npy"
        else:
            dataset_dir = root / dataset_name
            annotations = dataset_dir / "episodic_annotations"
            index_file = dataset_dir / "episode_frame_index.npz"
        annotations.mkdir(parents=True)

        index = {
            "index_frame_pair": np.array([[0, 4]], dtype=np.int64),
            "index_to_episode_id": np.array([episode_id], dtype=object),
        }
        if legacy:
            np.save(index_file, index, allow_pickle=True)
        else:
            np.savez(index_file, **index)
        np.save(
            annotations / f"{episode_id}.npy",
            {
                "caption": "pick up the object",
                "plan": "reach then grasp",
                "fps": 12,
                "obj_class": "fixture_task",
                "video_name": "test/fixture_task/video_001",
            },
            allow_pickle=True,
        )
        return annotations, episode_id

    def _assert_layout_pipeline(self, *, legacy: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_name = "custom-legacy-release" if legacy else "custom-hf-release"
            annotations, episode_id = self._write_layout(root, dataset_name, legacy=legacy)

            layout = resolve_dataset_layout(root, dataset_name)
            self.assertEqual(layout.layout_format, "legacy" if legacy else "hf")
            self.assertEqual(layout.annotations_dir, annotations)
            index = load_dataset_index(layout)
            self.assertEqual(index["index_frame_pair"].tolist(), [[0, 4]])
            self.assertEqual(index["index_to_episode_id"].tolist(), [episode_id])

            episode = load_episode_metadata(layout, episode_id)
            self.assertEqual(episode["instruction"], "pick up the object")
            self.assertEqual(episode["cot"], "reach then grasp")

            with mock.patch.dict(os.environ, {}, clear=True):
                dataset = FrameDataset(
                    dataset_folder=root,
                    dataset_name=dataset_name,
                    augmentation=False,
                    normalization=False,
                    load_images=False,
                    state_mask_prob=0.0,
                    target_fps=12,
                    source_fps=12,
                    filter_samples_by_fps=False,
                )
            self.assertEqual(Path(dataset.episodic_dataset_core.label_folder), annotations)
            self.assertEqual(dataset.episodic_dataset_core.index_frame_pair.tolist(), [[0, 4]])
            runtime_episode = dataset.episodic_dataset_core._load_episode_npy(
                str(annotations / f"{episode_id}.npy")
            )
            self.assertEqual(runtime_episode["instruction"], "pick up the object")
            self.assertEqual(runtime_episode["cot"], "reach then grasp")

            eval_episode = eval_load_episode(root, dataset_name, episode_id)
            self.assertEqual(eval_episode["instruction"], "pick up the object")
            self.assertEqual(training_task_counts(root, dataset_name), {"fixture_task": 1})
            mapping, counts = summarize_test_episode_tasks(root, dataset_name)
            self.assertEqual(mapping, {episode_id: "fixture_task"})
            self.assertEqual(counts, {"fixture_task": 1})

    def test_hf_layout_crosses_frame_statistics_and_eval_loaders(self):
        self._assert_layout_pipeline(legacy=False)

    def test_generic_legacy_layout_crosses_frame_statistics_and_eval_loaders(self):
        self._assert_layout_pipeline(legacy=True)

    def test_versioned_and_legacy_sidecar_branches_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            versioned = root / "versioned.json"
            versioned.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "plans": {"episode_1:4": {"plan": "reach and close"}},
                    }
                ),
                encoding="utf-8",
            )
            legacy = root / "legacy.json"
            legacy.write_text(json.dumps({"episode_1:4": "legacy plan"}), encoding="utf-8")

            self.assertEqual(
                load_predicted_plan_sidecar(versioned), {"episode_1:4": "reach and close"}
            )
            self.assertEqual(
                load_predicted_plan_sidecar(legacy), {"episode_1:4": "legacy plan"}
            )
            self.assertEqual(canonical_sample_id("episode_1", 4), "episode_1:4")

    def test_versioned_sidecar_requires_supported_version_and_plans_object(self):
        invalid_payloads = [
            ([], "JSON object"),
            ({"schema_version": "1.0"}, "plans object"),
            ({"plans": {"ep:0": "move"}}, "schema_version"),
            ({"schema_version": "2.0", "plans": {"ep:0": "move"}}, "schema_version"),
            ({"schema_version": 1.0, "plans": {"ep:0": "move"}}, "schema_version"),
            ({"schema_version": "1.0", "plans": []}, "JSON object"),
            ({"schema_version": "1.0", "plans": {}}, "at least one plan"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.json"
            for payload, message in invalid_payloads:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_predicted_plan_sidecar(path)

    def test_sidecar_rejects_noncanonical_ids_empty_plans_and_wrong_types(self):
        invalid_plans = [
            {"episode_1_4": "move"},
            {":4": "move"},
            {"episode_1:": "move"},
            {"episode_1:-1": "move"},
            {"episode_1:04": "move"},
            {"episode 1:4": "move"},
            {"episode/1:4": "move"},
            {"episode_1:4:5": "move"},
            {"episode_1:4": "   "},
            {"episode_1:4": 7},
            {"episode_1:4": {}},
            {"episode_1:4": {"plan": "nested legacy entries are not the raw-map ABI"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.json"
            for payload in invalid_plans:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_predicted_plan_sidecar(path)

            versioned_invalid_entries = [
                {"episode_1:4": ""},
                {"episode_1:4": {}},
                {"episode_1:4": {"plan": []}},
                {"episode_1:4": 7},
            ]
            for plans in versioned_invalid_entries:
                with self.subTest(plans=plans):
                    path.write_text(
                        json.dumps({"schema_version": "1.0", "plans": plans}),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        load_predicted_plan_sidecar(path)

    def test_duplicate_sample_ids_are_rejected_during_json_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.json"
            path.write_text(
                '{"schema_version":"1.0","plans":{"ep:0":"first","ep:0":"second"}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                load_predicted_plan_sidecar(path)

    def test_traversal_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                resolve_dataset_layout(directory, "../escape", require=False)


if __name__ == "__main__":
    unittest.main()
