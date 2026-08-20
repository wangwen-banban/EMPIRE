import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from empire.utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
MARKERS = [
    "/" + "home",
    "/" + "data",
    "/" + "mnt",
    "/" + "Users",
    "wang" + "wen",
    "snap" + "shots/",
]


class PublicConfigTest(unittest.TestCase):
    def test_release_configs_are_fresh_and_sanitized(self):
        configs = sorted((ROOT / "empire" / "configs").rglob("*.json"))
        self.assertEqual(len(configs), 14)
        for path in configs:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
            self.assertFalse(payload["resume"], path)
            self.assertEqual(payload["output_root"], "${OUTPUT_ROOT}")
            self.assertEqual(payload["train_dataset"]["data_root_dir"], "${DATA_ROOT}")
            self.assertEqual(payload["train_dataset"]["video_root_dir"], "${VIDEO_ROOT}")
            self.assertEqual(payload["vlm"]["pretrained_model_name_or_path"], "google/paligemma2-3b-mix-224")
            for marker in MARKERS:
                self.assertNotIn(marker, text, path)

    def test_reported_configuration_mapping(self):
        directory = ROOT / "empire" / "configs" / "ablations"
        load = lambda name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))

        self.assertFalse(load("baseline_caption_only")["train_dataset"]["use_cot"])
        self.assertTrue(load("plan_conditioning")["train_dataset"]["use_cot"])
        self.assertEqual(load("depth_cross_attention")["depth_image_encoder_type"], "CA")
        self.assertEqual(load("depth_concatenation")["depth_image_encoder_type"], "CONCAT")
        self.assertEqual(load("single_stage_joint_training")["train_setup"]["training_stage"], "stage_all")
        self.assertTrue(load("trainable_planner")["train_condition_encoders"])
        self.assertFalse(load("frozen_planner")["train_condition_encoders"])
        self.assertEqual(
            [load(name)["action_model"]["model_type"] for name in ("dit_small", "dit_base", "dit_large")],
            ["DiT-M", "DiT-B", "DiT-L"],
        )
        ground_truth = load("ground_truth_plan")
        predicted = load("predicted_plan")
        self.assertNotIn("cot_source", ground_truth["train_dataset"])
        self.assertEqual(predicted["train_dataset"]["cot_source"], "predicted")

        # Supplementary comparison: every architecture, data, schedule, batch,
        # and depth field is identical after removing only source-ABI fields.
        expected_ground_truth = copy.deepcopy(predicted)
        expected_ground_truth["task_name"] = "ground_truth_plan"
        expected_ground_truth["train_dataset"].pop("cot_source")
        expected_ground_truth["train_dataset"].pop("predicted_plan_sidecar")
        self.assertEqual(ground_truth, expected_ground_truth)
        for config in (ground_truth, predicted):
            self.assertEqual(config["action_model"]["model_type"], "DiT-B")
            self.assertEqual(config["action_model"]["training_objective"], "flow_matching")
            self.assertTrue(config["use_depth"])
            self.assertEqual(config["depth_image_encoder_type"], "CONCAT")
            self.assertEqual(config["batch_size"], 4)
            self.assertEqual(config["total_batch_size"], 64)
            self.assertEqual(config["fwd_pred_next_n"], 60)
            self.assertEqual(config["trainer"]["max_epochs"], 4)
            self.assertEqual(config["trainer"]["max_steps"], 40684)
            self.assertEqual(config["train_dataset"]["data_mix"], "empire_train")
            self.assertEqual(config["train_dataset"]["source_fps"], 30)
            self.assertEqual(config["train_dataset"]["target_fps"], 12)

        # Table 4 '+ motion plan' is the no-depth 20k pilot, not a renamed
        # supplementary full-data GT-plan recipe.
        plan = load("plan_conditioning")
        self.assertTrue(plan["cot_condition_span"])
        self.assertTrue(plan["train_dataset"]["use_cot"])
        self.assertFalse(plan["use_depth"])
        self.assertEqual(plan["action_model"]["condition_mode"], "vlm_cross_attention")
        self.assertEqual(plan["batch_size"], 8)
        self.assertEqual(plan["total_batch_size"], 64)
        self.assertEqual(plan["trainer"]["max_epochs"], 1)
        self.assertEqual(plan["trainer"]["max_steps"], 20000)
        self.assertNotEqual(plan["use_depth"], ground_truth["use_depth"])
        self.assertNotEqual(plan["trainer"]["max_steps"], ground_truth["trainer"]["max_steps"])

    def test_recursive_expansion_and_missing_variable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"nested": {"paths": ["${EMPIRE_TEST_ROOT}/a"]}}', encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(EnvironmentError, "EMPIRE_TEST_ROOT"):
                    load_config(path)
            with mock.patch.dict(os.environ, {"EMPIRE_TEST_ROOT": "relative-root"}, clear=True):
                config = load_config(path)
            self.assertEqual(config["nested"]["paths"], ["relative-root/a"])


if __name__ == "__main__":
    unittest.main()
