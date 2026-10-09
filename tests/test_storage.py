import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.storage import VideoStorage


class VideoStorageTest(unittest.TestCase):
    def test_upload_creates_numbered_directories_and_reuses_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            storage = VideoStorage(Path(temporary_directory) / "data")

            with patch("src.video_io.inspect_video") as inspect, patch(
                "src.storage.cv2.VideoCapture"
            ) as capture:
                inspect.return_value.to_dict.return_value = {
                    "width": 160,
                    "height": 120,
                    "fps": 10.0,
                    "frame_count": 10,
                    "duration_seconds": 1.0,
                }
                capture.return_value.read.return_value = (False, None)
                first, first_created = storage.save_upload(io.BytesIO(b"video-a"), "a.mp4")
                second, second_created = storage.save_upload(io.BytesIO(b"video-b"), "b.mov")
                duplicate, duplicate_created = storage.save_upload(
                    io.BytesIO(b"video-a"), "renamed.mp4"
                )

            self.assertTrue(first_created)
            self.assertTrue(second_created)
            self.assertFalse(duplicate_created)
            self.assertEqual(first.video_id, 1)
            self.assertEqual(second.video_id, 2)
            self.assertEqual(duplicate.video_id, first.video_id)
            self.assertEqual(first.file_path.read_bytes(), b"video-a")
            self.assertEqual(second.file_path.read_bytes(), b"video-b")
            self.assertTrue((first.directory / "metadata.json").is_file())
            self.assertEqual([video.video_id for video in storage.list_videos()], [2, 1])
            metadata = storage.update_metadata(
                first.video_id, {"lighting": "night", "intersection": "Cầu Giấy"}
            )
            self.assertEqual(metadata["lighting"], "night")
            self.assertEqual(storage.read_metadata(first.video_id)["intersection"], "Cầu Giấy")

    def test_rejects_unsupported_and_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            storage = VideoStorage(Path(temporary_directory) / "data")

            with self.assertRaisesRegex(ValueError, "Định dạng"):
                storage.save_upload(io.BytesIO(b"content"), "notes.txt")

            with self.assertRaisesRegex(ValueError, "rỗng"):
                storage.save_upload(io.BytesIO(b""), "empty.mp4")


if __name__ == "__main__":
    unittest.main()
