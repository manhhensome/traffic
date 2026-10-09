from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.analysis import AnalysisConfig, run_analysis


def write_job(path: Path, payload: dict) -> None:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: worker.py JOB_JSON")
    job_path = Path(sys.argv[1]).resolve()
    payload = json.loads(job_path.read_text(encoding="utf-8"))

    def progress(value: float, message: str) -> None:
        current = json.loads(job_path.read_text(encoding="utf-8"))
        current["progress"] = value
        current["message"] = message
        current["status"] = "processing"
        write_job(job_path, current)

    try:
        run_directory = run_analysis(
            payload["video_path"],
            payload["video_directory"],
            [tuple(point) for point in payload["roi_points"]],
            AnalysisConfig(**payload["config"]),
            progress,
        )
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        payload.update(
            status="completed",
            progress=1.0,
            message="Hoàn tất phân tích",
            run_directory=str(run_directory),
            error=None,
        )
    except Exception as error:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        payload.update(status="failed", message="Phân tích thất bại", error=str(error))
    write_job(job_path, payload)


if __name__ == "__main__":
    main()
