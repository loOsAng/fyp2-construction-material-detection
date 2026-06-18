import unittest
from pathlib import Path

from modules.model import list_available_models


class ModelSelectionTest(unittest.TestCase):
    def test_previous_v8x_balanced_model_is_visible_when_checkpoint_exists(self):
        checkpoint = Path("bestModelSelect/yolov8x_balanced/best.pt")
        self.assertTrue(checkpoint.exists(), "Expected previous YOLOv8x balanced checkpoint to exist")

        models = list_available_models()

        self.assertIn("YOLOv8x balanced", models)
        self.assertEqual(models["YOLOv8x balanced"], "bestModelSelect\\yolov8x_balanced\\best.pt")


if __name__ == "__main__":
    unittest.main()
