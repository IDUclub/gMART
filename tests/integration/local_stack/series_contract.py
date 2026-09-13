"""An incomplete or mixed-build batch cannot be called a successful full series."""


def verify_series_manifest(report):
    expected = (
        ["analysis"] * 10 + ["data"] * 4 + ["documents"] * 4 + ["continuation"] * 2
    )
    cases = report.get("cases", [])
    assert len(cases) == len(expected), "The fixed series requires all 20 cases"
    assert [case.get("index") for case in cases] == list(
        range(1, 21)
    ), "Case indices are incomplete or duplicated"
    assert [
        case.get("kind") for case in cases
    ] == expected, "Case kinds or order differ from the fixed fixture"
    assert len({case.get("request_id") for case in cases}) == 20 and all(
        case.get("request_id") for case in cases
    ), "Each case needs a distinct request ID"
    initial = report.get("fingerprint") or {}
    assert all(
        initial.get(key) for key in ("commit", "source_sha256", "agents_image")
    ), "Build fingerprint is incomplete"
    assert (
        report.get("final_fingerprint") == initial
    ), "Series did not finish on the original build"
