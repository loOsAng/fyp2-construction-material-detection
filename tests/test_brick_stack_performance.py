import unittest
from unittest.mock import patch

import numpy as np
import torch

from modules import skeleton


class BrickStackPerformanceTest(unittest.TestCase):
    def test_instance_isolation_erodes_each_mask_once(self):
        masks = []
        for idx in range(40):
            mask = np.zeros((96, 96), dtype=np.uint8)
            row, col = divmod(idx, 8)
            y = 2 + row * 10
            x = 2 + col * 10
            mask[y:y + 8, x:x + 8] = 1
            masks.append(mask)

        original_erode = skeleton.cv2.erode
        calls = 0

        def counting_erode(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_erode(*args, **kwargs)

        with patch.object(skeleton.cv2, "erode", side_effect=counting_erode):
            isolated = skeleton._isolate_by_erosion(masks)

        self.assertEqual(len(isolated), len(masks))
        self.assertLessEqual(calls, len(masks) + 1)

    def test_small_instance_refinement_uses_tight_crop(self):
        class Boxes:
            cls = torch.zeros(1, dtype=torch.float32)

        class Masks:
            data = torch.zeros((1, 128, 128), dtype=torch.float32)

        class Result:
            masks = Masks()
            boxes = Boxes()
            names = {0: "Brick"}

        Result.masks.data[0, 50:60, 70:80] = 1
        seen_shapes = []

        def fake_refine(binary, class_key=""):
            seen_shapes.append(binary.shape)
            return binary

        with patch.object(skeleton, "_refine_instance_mask", side_effect=fake_refine):
            binaries = skeleton._prepare_instance_binaries(Result(), 128, 128)

        self.assertEqual(binaries[0].shape, (128, 128))
        self.assertLess(seen_shapes[0][0], 128)
        self.assertLess(seen_shapes[0][1], 128)

    def test_dense_brick_stack_skips_global_isolation(self):
        class Boxes:
            cls = torch.zeros(90, dtype=torch.float32)

        class Masks:
            data = torch.zeros((90, 64, 64), dtype=torch.float32)

        class Result:
            masks = Masks()
            boxes = Boxes()
            names = {0: "Brick"}

        for idx in range(90):
            row, col = divmod(idx, 10)
            y = row * 6
            x = col * 6
            Result.masks.data[idx, y:y + 4, x:x + 4] = 1

        with patch.object(skeleton, "_isolate_by_erosion", side_effect=AssertionError("global isolation called")):
            binaries = skeleton._prepare_instance_binaries(Result(), 64, 64)

        self.assertEqual(len(binaries), 90)


if __name__ == "__main__":
    unittest.main()
