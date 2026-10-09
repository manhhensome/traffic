from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import psutil

from src.analysis import AnalysisConfig, recover_incomplete_runs


ACTIVE_STATUSES = {"queued", "processing"}
_PROCESSES: dict[int, subprocess.Popen] = {}


def _write_job(path: Path, payload: dict) -> None:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_job(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pid = payload.get("pid")
    process = _PROCESSES.get(int(pid)) if pid else None
    if process is not None and process.poll() is not None:
        _PROCESSES.pop(int(pid), None)
    return payload


def list_jobs(video_directory: str | Path) -> list[Path]:
    jobs_directory = Path(video_directory) / "jobs"
    return sorted(jobs_directory.glob("job-*.json"), reverse=True) if jobs_directory.is_dir() else []


def latest_job(video_directory: str | Path) -> tuple[Path, dict] | None:
    for path in list_jobs(video_directory):
        try:
            return path, read_job(path)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def find_active_job(data_directory: str | Path) -> tuple[Path, dict] | None:
    for path in Path(data_directory).glob("video-*/jobs/job-*.json"):
        try:
            payload = read_job(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") in ACTIVE_STATUSES:
            pid = payload.get("pid")
            if pid and psutil.pid_exists(int(pid)):
                return path, payload
            payload["status"] = "interrupted"
            payload["error"] = "Worker không còn hoạt động"
            _write_job(path, payload)
    return None


def start_analysis_job(
    project_directory: str | Path,
    video_path: str | Path,
    video_directory: str | Path,
    roi_points: list[tuple[int, int]],
    config: AnalysisConfig,
) -> Path:
    project_path = Path(project_directory).resolve()
    video_directory_path = Path(video_directory).resolve()
    if find_active_job(project_path / "data") is not None:
        raise RuntimeError("Đang có một tác vụ phân tích khác. Hãy chờ hoặc hủy tác vụ đó.")
    jobs_directory = video_directory_path / "jobs"
    jobs_directory.mkdir(exist_ok=True)
    job_path = jobs_directory / f"job-{uuid4().hex}.json"
    payload = {
        "schema_version": 1,
        "status": "queued",
        "progress": 0.0,
        "message": "Đang chờ worker",
        "video_path": str(Path(video_path).resolve()),
        "video_directory": str(video_directory_path),
        "roi_points": roi_points,
        "config": asdict(config),
        "pid": None,
        "run_directory": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_job(job_path, payload)
    payload["status"] = "processing"
    payload["message"] = "Worker đang xử lý video"
    _write_job(job_path, payload)
    log_path = job_path.with_suffix(".log")
    with log_path.open("ab") as log_file:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [sys.executable, str(project_path / "worker.py"), str(job_path)],
            cwd=project_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
    current = read_job(job_path)
    current["pid"] = process.pid
    _write_job(job_path, current)
    _PROCESSES[process.pid] = process
    return job_path


def cancel_job(job_path: str | Path) -> dict:
    path = Path(job_path)
    payload = read_job(path)
    pid = payload.get("pid")
    if pid and psutil.pid_exists(int(pid)):
        process = psutil.Process(int(pid))
        process.terminate()
        try:
            process.wait(timeout=5)
        except psutil.TimeoutExpired:
            process.kill()
    payload["status"] = "interrupted"
    payload["message"] = "Tác vụ đã được hủy"
    payload["error"] = "Người dùng hủy tác vụ"
    _write_job(path, payload)
    recover_incomplete_runs(Path(payload["video_directory"]).parent)
    return payload

