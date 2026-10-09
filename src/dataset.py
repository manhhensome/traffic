from __future__ import annotations

import json
import random
from pathlib import Path


VALID_SPLITS = {"train", "val", "test"}


def load_splits(data_directory: str | Path) -> dict[str, str]:
    path = Path(data_directory) / "splits.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): value for key, value in payload.get("videos", {}).items()}


def save_split(data_directory: str | Path, video_id: int, split: str) -> dict[str, str]:
    if split not in VALID_SPLITS:
        raise ValueError("Split phải là train, val hoặc test.")
    data_path = Path(data_directory)
    assignments = load_splits(data_path)
    assignments[str(video_id)] = split
    payload = {"schema_version": 1, "videos": assignments}
    (data_path / "splits.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return assignments


def create_grouped_splits(
    data_directory: str | Path,
    metadata_items: list[dict],
    seed: int = 42,
) -> dict[str, str]:
    groups: dict[str, list[int]] = {}
    for metadata in metadata_items:
        group_name = metadata.get("recording_session_id") or f"video-{metadata['video_id']}"
        groups.setdefault(str(group_name), []).append(int(metadata["video_id"]))

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    total_videos = max(1, sum(len(video_ids) for _, video_ids in group_items))
    assignments: dict[str, str] = {}
    assigned = 0
    for _, video_ids in group_items:
        ratio = assigned / total_videos
        split = "train" if ratio < 0.60 else "val" if ratio < 0.80 else "test"
        for video_id in video_ids:
            assignments[str(video_id)] = split
        assigned += len(video_ids)

    payload = {
        "schema_version": 1,
        "seed": seed,
        "strategy": "grouped_by_recording_session_60_20_20",
        "videos": assignments,
    }
    data_path = Path(data_directory)
    (data_path / "splits.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return assignments

