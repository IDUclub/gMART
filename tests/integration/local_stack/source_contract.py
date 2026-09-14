"""Semantic oracle for the synthetic fixture, independent of model prose."""

CLAUSE_TEXT = (
    "Расстояние от здания школы до открытой автомобильной стоянки "
    "должно быть не менее 50 метров."
)


def verify_source_records(records):
    clauses = [
        source
        for source in records.get("documents", [])
        if source.get("id")
        and source.get("doc_id")
        and source.get("name") == "LOCAL SDK TEST"
        and source.get("version") == "2026"
        and source.get("version_id")
        and source.get("numbering") == "1.1"
        and " ".join(source.get("text", "").split()) == CLAUSE_TEXT
    ]
    assert clauses, "Missing the original synthetic clause 1.1 and its version"
    for rule in records.get("norms", []):
        provenance = rule.get("provenance") or {}
        value = rule.get("value") or {}
        if not (
            rule.get("id")
            and "школ" in rule.get("subject", "").casefold()
            and "стоян" in rule.get("object", "").casefold()
            and value.get("number") == 50
            and value.get("operator") == ">="
            and value.get("unit") in {"м", "m"}
        ):
            continue
        for clause in clauses:
            if provenance.get("clause_node_id") == clause["id"] and all(
                provenance.get(field) == clause[field]
                for field in ("doc_id", "name", "version", "version_id")
            ):
                return
    raise AssertionError(
        "No matching school/parking restriction >= 50 m from clause 1.1"
    )
