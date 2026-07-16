from types import SimpleNamespace

from lexical_index import lexical_query_lanes, query_tokens, replace_source, search


def test_exact_identifier_is_retrievable(tmp_path) -> None:
    database = tmp_path / "lexical.sqlite3"
    document = SimpleNamespace(
        page_content="Alarm code TRX-991 requires a controller reset.",
        metadata={
            "source_id": "manual.pdf",
            "document_title": "Manual",
            "section_path": "Alarms",
        },
    )
    replace_source(database, "manual.pdf", [document], ["chunk-1"])
    results = search(database, "TRX-991", 5)
    assert results[0][0] == "chunk-1"


def test_query_glue_is_removed_in_english_and_portuguese() -> None:
    assert query_tokens("What does the document define as maximum speed?") == [
        "maximum",
        "speed",
    ]
    assert query_tokens("Qual é a definição de limite de uma função?") == [
        "limite",
        "função",
    ]


def test_compact_technical_query_gets_phrase_and_keyword_lanes() -> None:
    lanes = lexical_query_lanes("Station Dwell Reaction Time")
    assert lanes[0][0] == '"station dwell reaction time"'
    assert any(" OR " in query for query, _ in lanes)


def test_repeated_document_title_does_not_beat_matching_content(tmp_path) -> None:
    database = tmp_path / "lexical.sqlite3"
    documents = [
        SimpleNamespace(
            page_content="This section discusses maintenance staffing.",
            metadata={
                "source_id": "manual.pdf",
                "document_title": "Monorail operating manual",
                "section_path": "Staffing",
            },
        ),
        SimpleNamespace(
            page_content="The maximum operational speed is 80 km/h.",
            metadata={
                "source_id": "manual.pdf",
                "document_title": "Monorail operating manual",
                "section_path": "Key operating requirements",
            },
        ),
    ]
    replace_source(database, "manual.pdf", documents, ["irrelevant", "speed"])
    results = search(database, "What is the maximum operational speed?", 5)
    assert results[0][0] == "speed"
