import subprocess
import sys
import unittest
from pathlib import Path


class WheelImportSmokeTest(unittest.TestCase):
    def test_base_package_imports_without_eager_model_loading(self):
        code = "import empire; assert empire.__version__ == '0.1.0'"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_public_extra_covers_dataset_mesh_and_da3_imports(self):
        code = (
            "import utils3d, trimesh; "
            "import empire.datasets.augment_utils; "
            "import empire.utils.mesh_utils; "
            "import empire.models.depth_anything_3.utils.pose_align"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_training_entrypoint_imports_in_public_environment(self):
        train_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
        code = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('train_import_smoke', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module)"
        )
        subprocess.run([sys.executable, "-c", code, str(train_path)], check=True)


if __name__ == "__main__":
    unittest.main()
