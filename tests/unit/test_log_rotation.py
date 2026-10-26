import gzip

import pytest

from src.agents.common.logging import log_rotation as agents_rotation
from src.idu_mcp.common.logging import log_rotation as mcp_rotation


@pytest.fixture(params=[agents_rotation, mcp_rotation], ids=["agents", "idu_mcp"])
def rotation(request):
    return request.param


def test_small_log_is_left_alone(tmp_path, rotation):
    log = tmp_path / "app.log"
    log.write_bytes(b"x" * 10)

    archive = rotation.rotate_log_if_needed(log, rotate_bytes=100, max_total_bytes=1000)

    assert archive is None
    assert log.read_bytes() == b"x" * 10
    assert not list(tmp_path.glob("app.log.*.gz"))


def test_large_log_is_archived_and_truncated(tmp_path, rotation):
    log = tmp_path / "app.log"
    content = b"line\n" * 50
    log.write_bytes(content)

    archive = rotation.rotate_log_if_needed(
        log, rotate_bytes=100, max_total_bytes=10_000
    )

    assert archive is not None
    assert gzip.decompress(archive.read_bytes()) == content
    assert log.read_bytes() == b""


def test_writer_keeps_appending_after_truncation(tmp_path, rotation):
    log = tmp_path / "app.log"
    with open(log, "a", encoding="utf-8") as writer:
        writer.write("old\n" * 50)
        writer.flush()

        rotation.rotate_log_if_needed(log, rotate_bytes=100, max_total_bytes=10_000)
        writer.write("new\n")
        writer.flush()

    assert log.read_text(encoding="utf-8") == "new\n"


def test_oldest_archives_are_pruned_to_fit_the_limit(tmp_path, rotation):
    log = tmp_path / "app.log"
    log.write_bytes(b"")
    for stamp in ("20260101-000000", "20260102-000000", "20260103-000000"):
        (tmp_path / f"app.log.{stamp}.gz").write_bytes(b"z" * 40)

    rotation.rotate_log_if_needed(log, rotate_bytes=100, max_total_bytes=190)

    remaining = sorted(path.name for path in tmp_path.glob("app.log.*.gz"))
    assert remaining == ["app.log.20260102-000000.gz", "app.log.20260103-000000.gz"]
