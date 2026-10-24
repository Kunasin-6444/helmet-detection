"""Run with unittest discovery. No trained weights or Ultralytics required."""

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from detector import annotate, load_model, process_video, validate_names


def fake_result():
    # Deliberately reverse IDs to catch code relying on the original class order.
    box = SimpleNamespace(cls=np.array([0]), conf=np.array([0.9]),
                          xyxy=np.array([[30, 35, 90, 100]]))
    return SimpleNamespace(names={0: "helmet", 1: "head"}, boxes=[box])


class FakeModel:
    def predict(self, **kwargs):
        return [fake_result()]


class PipelineChecks(unittest.TestCase):
    def test_class_mapping_and_png(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        result, counts = annotate(frame, fake_result())
        self.assertEqual(counts, {"helmet": 1, "head": 0})
        self.assertFalse(frame.any(), "Must not draw on the original input")
        self.assertTrue(result.any())
        ok, png = cv2.imencode(".png", result)
        self.assertTrue(ok)
        self.assertEqual(cv2.imdecode(png, cv2.IMREAD_COLOR).shape, frame.shape)

    def test_empty_detections(self):
        empty = SimpleNamespace(names={0: "head", 1: "helmet"}, boxes=[])
        _, counts = annotate(np.zeros((120, 180, 3), dtype=np.uint8), empty)
        self.assertEqual(counts, {"helmet": 0, "head": 0})

    def test_wrong_model(self):
        with self.assertRaises(ValueError):
            validate_names({0: "person"})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_model(Path(folder) / "missing.pt")

    def test_video_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "input.mp4", Path(folder) / "result.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                                     10, (180, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((120, 180, 3), dtype=np.uint8))
            writer.release()
            info = process_video(FakeModel(), source, output, 0.25)
            self.assertEqual(info["frames"], 5)
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertTrue(frame.any())
            finally:
                capture.release()

    def test_invalid_video(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "broken.mp4"
            source.write_bytes(b"not a video")
            with self.assertRaises(ValueError):
                process_video(FakeModel(), source, Path(folder) / "result.mp4", 0.25)


if __name__ == "__main__":
    unittest.main()
