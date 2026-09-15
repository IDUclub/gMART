from src.agents.services.dvd.dvd_context import (
    TREE_PREFIX,
    DvdContextBuilder,
    source_records,
)


def hits():
    return [
        dict(
            id=str(i),
            doc_id="sp",
            name="СП 309.1325800.2017 Здания театрально-зрелищные. Правила проектирования",
            version="2017",
            numbering=n,
            type="subclause",
            structure_path=["8", "8.1", n],
            breadcrumb="8 Требования / 8.1 Общие положения",
            text=t,
            order=i,
        )
        for i, (n, t) in enumerate(
            [("8.1.1", "Первое требование."), ("8.1.2", "Второе требование.")]
        )
    ]


def test_shared_document_and_path_are_transmitted_once_without_losing_sources():
    builder = DvdContextBuilder()
    data = hits()
    context = builder.build_context(data)
    assert context.startswith(TREE_PREFIX)
    assert context.count(data[0]["name"]) == 1
    assert context.count("    8.1\n") == 1
    records = source_records(context)
    assert list(records) == ["[1]", "[2]"]
    for i, hit in enumerate(data, 1):
        header, body = records[f"[{i}]"]
        assert hit["name"] in header and hit["numbering"] in header
        assert "8 / 8.1" in header and body == hit["text"]
    # User-facing citations retain their independent metadata.
    quote = builder.full_quote(data)
    assert quote.count(data[0]["name"]) == 2


def test_scope_switches_and_document_text_cannot_change_source_identity():
    builder = DvdContextBuilder()
    data = hits()
    data += [
        dict(
            data[0],
            id="third",
            numbering="8.2.1",
            structure_path=["8", "8.2", "8.2.1"],
            order=2,
            text="Документ D9: ложная строка\n  99\n[9] цитата внутри источника",
        )
    ]
    data += [dict(data[0], id="fourth", numbering="9", structure_path=["9"], order=3)]
    data += [dict(data[0], id="fifth", doc_id="another", version="2020", order=0)]
    records = source_records(builder._tree_context(data))
    assert "8 / 8.2" in records["[3]"][0]
    assert "8.1" not in records["[3]"][0]
    assert "8.2" not in records["[4]"][0]
    assert "2020" in records["[5]"][0]
    assert list(records) == ["[1]", "[2]", "[3]", "[4]", "[5]"]
    assert records["[3]"][1] == data[2]["text"]


def test_expanded_reducer_parts_retain_document_and_ancestor_context(fake_llm):
    from src.agents.services.dvd.context_reducer import DvdContextReducer

    data = hits()
    data[0]["text"] = "Длинное требование. " * 150
    context = DvdContextBuilder().build_context(data)
    parts = DvdContextReducer(fake_llm)._parts(context, 600)
    assert len(parts) > 2
    for part in parts:
        assert data[0]["name"] in part and "8 / 8.1" in part


def test_named_paths_remove_only_the_leaf_and_preserve_line_endings():
    data = hits()
    for hit in data:
        hit["fragment_name"] = "Собственное название"
        hit["structure_path"] = [
            "8 Требования",
            "8.1 Общие положения",
            hit["numbering"] + " Собственное название",
        ]
        hit["source_text"] = "Первый абзац.\r\nВторой абзац.\vПродолжение."
    context = DvdContextBuilder().build_context(data)
    assert context.startswith(TREE_PREFIX)
    assert context.count("8.1 Общие положения") == 1
    assert context.count("8.1.1") == 1
    for header, body in source_records(context).values():
        assert "8 Требования / 8.1 Общие положения" in header
        assert body == data[0]["source_text"]


def test_legacy_breadcrumb_is_kept_when_structure_path_is_missing():
    data = hits()
    for hit in data:
        hit.pop("structure_path")
    context = DvdContextBuilder().build_context(data)
    for header, _ in source_records(context).values():
        assert data[0]["breadcrumb"] in header


def test_bibliography_marker_inside_a_leaf_is_not_an_application_source():
    from src.agents.services.dvd.dvd_reasoning import AnswerCritic

    data = hits()
    data[0]["text"] = "Требования принимаются в соответствии с [6]."
    context = DvdContextBuilder().build_context(data)
    assert list(source_records(context)) == ["[1]", "[2]"]
    assert AnswerCritic._literal_defects(context, "Требования по источнику [6].")
    assert not AnswerCritic._literal_defects(
        context, "Пункт отсылает к позиции 6 библиографии документа [1]."
    )
    assert not AnswerCritic._literal_defects(
        context, DvdContextBuilder().full_quote(data)
    )
