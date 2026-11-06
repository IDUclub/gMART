import pytest


@pytest.fixture(autouse=True)
def isolated_compliance_cache(tmp_path, monkeypatch):
    """Request ids may repeat across hermetic tests, disk checkpoints must not."""
    monkeypatch.setenv("GMART_CACHE_DIR", str(tmp_path / "cache"))
