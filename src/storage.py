from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator
from uuid import uuid4

import cv2


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class StoredVideo:
    video_id: int
    directory: Path
    file_path: Path
    original_filename: str
    sha256: str
    size_bytes: int
    uploaded_at: str

    @property
    def label(self) -> str:
        size_mb = self.size_bytes / (1024 * 1024)
        return f"video-{self.video_id} · {self.original_filename} · {size_mb:.1f} MB"


class VideoStorage:
    """Persist uploaded videos in data/video-N directories."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.data_dir / "catalog.sqlite"
        self._initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _extension(filename: str) -> str:
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise ValueError(f"Định dạng video không được hỗ trợ. Hãy chọn: {allowed}")
        return extension

    def save_upload(self, uploaded_file: BinaryIO, original_filename: str) -> tuple[StoredVideo, bool]:
        extension = self._extension(original_filename)
        temporary_path = self.data_dir / f".uploading-{uuid4().hex}{extension}"
        digest = hashlib.sha256()
        size_bytes = 0

        uploaded_file.seek(0)
        try:
            with temporary_path.open("wb") as destination:
                while chunk := uploaded_file.read(1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)

            if size_bytes == 0:
                raise ValueError("Video rỗng, không thể lưu.")

            from src.video_io import inspect_video

            video_info = inspect_video(temporary_path)

            sha256 = digest.hexdigest()
            existing = self.get_by_hash(sha256)
            if existing is not None:
                return existing, False

            uploaded_at = datetime.now(timezone.utc).isoformat()
            stored_filename = f"original{extension}"

            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO videos (
                        original_filename, stored_filename, sha256,
                        size_bytes, uploaded_at, status
                    ) VALUES (?, ?, ?, ?, ?, 'uploading')
                    """,
                    (original_filename, stored_filename, sha256, size_bytes, uploaded_at),
                )
                video_id = int(cursor.lastrowid)

            video_directory = self.data_dir / f"video-{video_id}"
            try:
                video_directory.mkdir(parents=False, exist_ok=False)
                final_path = video_directory / stored_filename
                temporary_path.replace(final_path)

                metadata = {
                    "schema_version": 1,
                    "video_id": video_id,
                    "original_filename": original_filename,
                    "stored_filename": stored_filename,
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "uploaded_at": uploaded_at,
                    "status": "ready",
                    "recorded_at": None,
                    "source_type": "upload",
                    "source_url": None,
                    "intersection": None,
                    "camera_id": None,
                    "recording_session_id": None,
                    "lighting": "unknown",
                    "weather": "unknown",
                    **video_info.to_dict(),
                }
                (video_directory / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                capture = cv2.VideoCapture(str(final_path))
                success, thumbnail = capture.read()
                capture.release()
                if success and thumbnail is not None:
                    max_width = 640
                    if thumbnail.shape[1] > max_width:
                        scale = max_width / thumbnail.shape[1]
                        thumbnail = cv2.resize(
                            thumbnail,
                            (max_width, round(thumbnail.shape[0] * scale)),
                        )
                    cv2.imwrite(str(video_directory / "thumbnail.jpg"), thumbnail)
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE videos SET status = 'ready' WHERE video_id = ?",
                        (video_id,),
                    )
            except Exception:
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE videos SET status = 'failed' WHERE video_id = ?",
                        (video_id,),
                    )
                raise

            return self.get(video_id), True
        finally:
            temporary_path.unlink(missing_ok=True)

    def read_metadata(self, video_id: int) -> dict:
        video = self.get(video_id)
        return json.loads((video.directory / "metadata.json").read_text(encoding="utf-8"))

    def update_metadata(self, video_id: int, updates: dict) -> dict:
        protected = {
            "schema_version",
            "video_id",
            "original_filename",
            "stored_filename",
            "sha256",
            "size_bytes",
            "uploaded_at",
            "status",
            "width",
            "height",
            "fps",
            "frame_count",
            "duration_seconds",
        }
        metadata = self.read_metadata(video_id)
        metadata.update({key: value for key, value in updates.items() if key not in protected})
        video = self.get(video_id)
        temporary_path = video.directory / ".metadata.tmp"
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(video.directory / "metadata.json")
        return metadata

    def get(self, video_id: int) -> StoredVideo:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM videos WHERE video_id = ? AND status = 'ready'",
                (video_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Không tìm thấy video-{video_id}")
        return self._row_to_video(row)

    def get_by_hash(self, sha256: str) -> StoredVideo | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM videos WHERE sha256 = ? AND status = 'ready'",
                (sha256,),
            ).fetchone()
        return self._row_to_video(row) if row is not None else None

    def list_videos(self) -> list[StoredVideo]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM videos WHERE status = 'ready' ORDER BY video_id DESC"
            ).fetchall()
        videos = [self._row_to_video(row) for row in rows]
        return [video for video in videos if video.file_path.is_file()]

    def _row_to_video(self, row: sqlite3.Row) -> StoredVideo:
        directory = self.data_dir / f"video-{row['video_id']}"
        return StoredVideo(
            video_id=int(row["video_id"]),
            directory=directory,
            file_path=directory / row["stored_filename"],
            original_filename=row["original_filename"],
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            uploaded_at=row["uploaded_at"],
        )

