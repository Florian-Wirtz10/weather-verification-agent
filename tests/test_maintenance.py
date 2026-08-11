from __future__ import annotations

import os

from webapi.maintenance import cleanup_job_outputs


def _job(root, job_id: str, size: int, modified: float):
    directory = root / job_id
    directory.mkdir()
    (directory / "artifact.bin").write_bytes(b"x" * size)
    os.utime(directory, (modified, modified))
    return directory


def test_cleanup_removes_expired_jobs_but_preserves_active_and_unknown(tmp_path):
    now = 10 * 86_400
    expired = _job(tmp_path, "a" * 12, 12, now - 8 * 86_400)
    active = _job(tmp_path, "b" * 12, 13, now - 8 * 86_400)
    fresh = _job(tmp_path, "c" * 12, 14, now - 2 * 86_400)
    unknown = _job(tmp_path, "not-a-job", 15, now - 20 * 86_400)

    report = cleanup_job_outputs(
        tmp_path,
        retention_days=7,
        max_storage_gb=0,
        active_job_ids={active.name},
        now=now,
    )

    assert report.removed_job_ids == (expired.name,)
    assert report.bytes_freed == 12
    assert not expired.exists()
    assert active.exists()
    assert fresh.exists()
    assert unknown.exists()


def test_cleanup_enforces_quota_from_oldest_non_active_job(tmp_path):
    old = _job(tmp_path, "1" * 12, 700, 100)
    middle = _job(tmp_path, "2" * 12, 600, 200)
    newest = _job(tmp_path, "3" * 12, 500, 300)

    report = cleanup_job_outputs(
        tmp_path,
        retention_days=0,
        max_storage_gb=1_100 / 1024**3,
        active_job_ids={middle.name},
        now=400,
    )

    assert report.removed_job_ids == (old.name,)
    assert not old.exists()
    assert middle.exists()
    assert newest.exists()
    assert report.remaining_bytes == 1_100
