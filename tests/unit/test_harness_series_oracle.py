from copy import deepcopy

import pytest

from tests.integration.local_stack.series_contract import verify_series_manifest


def manifest():
    fingerprint = {
        "commit": "commit",
        "source_sha256": "source",
        "agents_image": "image",
    }
    kinds = ["analysis"] * 10 + ["data"] * 4 + ["documents"] * 4 + ["continuation"] * 2
    return {
        "fingerprint": fingerprint,
        "final_fingerprint": deepcopy(fingerprint),
        "cases": [
            {"index": i, "kind": kind, "request_id": str(i)}
            for i, kind in enumerate(kinds, 1)
        ],
    }


def test_complete_same_build_manifest_is_accepted():
    verify_series_manifest(manifest())


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "partial",
        "duplicate_index",
        "wrong_kind",
        "duplicate_request",
        "changed_build",
        "missing_finish",
    ],
)
def test_incomplete_or_mixed_series_cannot_pass(defect):
    report = manifest()
    if defect == "empty":
        report["cases"] = []
    elif defect == "partial":
        report["cases"].pop()
    elif defect == "duplicate_index":
        report["cases"][-1]["index"] = 1
    elif defect == "wrong_kind":
        report["cases"][-1]["kind"] = "data"
    elif defect == "duplicate_request":
        report["cases"][-1]["request_id"] = "1"
    elif defect == "changed_build":
        report["final_fingerprint"]["source_sha256"] = "changed"
    elif defect == "missing_finish":
        report.pop("final_fingerprint")
    with pytest.raises(AssertionError):
        verify_series_manifest(report)
