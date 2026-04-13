"""In-memory job store for avatar generation."""
import uuid
from dataclasses import dataclass
from typing import Optional, Any

class JobStatus:
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"

@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: int = 0
    progress_message: Optional[str] = None
    avatar_url: Optional[str] = None
    avatar_style: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Optional[dict[str, Any]] = None
    retryable: Optional[bool] = None
    request_id: Optional[str] = None
    face_path: Optional[str] = None
    body_front_path: Optional[str] = None
    body_side_path: Optional[str] = None
    body_debug: Optional[dict[str, Any]] = None

_jobs: dict = {}

def create_job(
    face_path: str,
    body_front_path: str,
    body_side_path: Optional[str] = None,
    job_id: Optional[str] = None,
    avatar_style: Optional[str] = None,
    request_id: Optional[str] = None,
) -> str:
    if job_id is None:
        job_id = str(uuid.uuid4())
    _jobs[job_id] = Job(
        id=job_id,
        face_path=face_path,
        body_front_path=body_front_path,
        body_side_path=body_side_path,
        avatar_style=avatar_style,
        request_id=request_id,
    )
    return job_id

def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)

def update_job(job_id: str, **kwargs) -> None:
    job = _jobs.get(job_id)
    if job:
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
