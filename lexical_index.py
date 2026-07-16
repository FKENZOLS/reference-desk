"""SQLite FTS5 sidecar used for exact-term and acronym retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Sequence


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*", re.UNICODE)
QUOTED_PHRASE_PATTERN = re.compile(r'["“”]([^"“”]{2,120})["“”]')

# These are deliberately limited to query glue and search-intent words. Terms
# that may carry technical meaning remain searchable. Both English and
# Portuguese are included because this corpus is queried in both languages.
STOPWORDS = frozenset(
    """
    a an and are as at be been by can could did do does for from had has have
    how i in into is it its may of on or please show that the their there these
    this those to was were what when where which who why will with would you
    document documents file files manual page pages find search tell give
    define defined definition explain locate according
    a ao aos aquela aquelas aquele aqueles com como da das de dela delas dele
    deles do dos e em entre essa essas esse esses esta estas este estes foi
    foram há isso isto já lhe lhes mais mas me minha minhas meu meus na nas no
    nos o os ou para pela pelas pelo pelos por qual quais que quem se sem ser
    seu seus sua suas são tem uma umas um uns você vocês documento documentos
    arquivo arquivos manual página páginas pagina paginas encontre encontrar
    buscar busca mostre mostrar diga dizer dê de acordo definição definicao
    definir explique explicar localize localizar
    """.split()
)

LEXICAL_RRF_K = 20


def fingerprint_ids(ids: Iterable[str]) -> str:
    payload = json.dumps(sorted(str(value) for value in ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            source_id UNINDEXED,
            title,
            section,
            content,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS index_state(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    return connection


def _row_for_document(document: Any, chunk_id: str) -> tuple[str, str, str, str, str]:
    metadata = document.metadata
    return (
        chunk_id,
        str(metadata.get("source_id", "")),
        str(metadata.get("document_title", "")),
        str(metadata.get("section_path", "")),
        document.page_content,
    )


def replace_source(
    path: Path,
    source_id: str,
    documents: Sequence[Any],
    ids: Sequence[str],
) -> None:
    if len(documents) != len(ids):
        raise ValueError("Lexical documents and IDs must have equal lengths.")

    rows = [
        _row_for_document(document, chunk_id)
        for document, chunk_id in zip(documents, ids, strict=True)
    ]
    with closing(connect(path)) as connection, connection:
        connection.execute(
            "DELETE FROM chunks_fts WHERE source_id = ?",
            (source_id,),
        )
        connection.executemany(
            """
            INSERT INTO chunks_fts(chunk_id, source_id, title, section, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def delete_source(path: Path, source_id: str) -> None:
    with closing(connect(path)) as connection, connection:
        connection.execute(
            "DELETE FROM chunks_fts WHERE source_id = ?",
            (source_id,),
        )


def query_tokens(question: str) -> list[str]:
    """Return unique meaningful terms while retaining technical identifiers."""

    all_tokens: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_PATTERN.findall(question.casefold()):
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        all_tokens.append(token)

    # A query made only of stopwords is unusual, but returning no lane at all
    # is less useful than falling back to the original non-trivial tokens.
    if not all_tokens:
        for token in TOKEN_PATTERN.findall(question.casefold()):
            if len(token) >= 2 and token not in seen:
                seen.add(token)
                all_tokens.append(token)
    return all_tokens


def quote_fts(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def lexical_query(question: str) -> str:
    """Build the broad keyword lane retained for API compatibility."""

    return " OR ".join(quote_fts(token) for token in query_tokens(question))


def lexical_query_lanes(question: str) -> list[tuple[str, float]]:
    """Build complementary phrase, identifier, and keyword FTS lanes."""

    terms = query_tokens(question)
    if not terms:
        return []

    lanes: list[tuple[str, float]] = []
    explicit_phrases = [
        " ".join(query_tokens(match.group(1)))
        for match in QUOTED_PHRASE_PATTERN.finditer(question)
    ]
    for phrase in explicit_phrases:
        if phrase:
            lanes.append((quote_fts(phrase), 3.0))

    # A compact meaningful query is often a definition, requirement name, or
    # technical phrase. If the exact phrase is absent, the broad lane below
    # still supplies ordinary BM25 matches.
    if 2 <= len(terms) <= 6:
        lanes.append((quote_fts(" ".join(terms)), 2.0))

    original_tokens = TOKEN_PATTERN.findall(question)
    identifiers = []
    for original in original_tokens:
        normalized = original.casefold()
        is_identifier = (
            any(character.isdigit() for character in original)
            or "-" in original
            or (len(original) >= 2 and original.isupper())
        )
        if is_identifier and normalized in terms and normalized not in identifiers:
            identifiers.append(normalized)
    if identifiers:
        lanes.append((" OR ".join(quote_fts(value) for value in identifiers), 2.5))

    lanes.append((" OR ".join(quote_fts(term) for term in terms), 1.0))

    deduplicated: list[tuple[str, float]] = []
    seen_queries: set[str] = set()
    for query, weight in lanes:
        if query not in seen_queries:
            seen_queries.add(query)
            deduplicated.append((query, weight))
    return deduplicated


def search(
    path: Path,
    question: str,
    limit: int,
    source_id: str | None = None,
) -> list[tuple[str, float]]:
    lanes = lexical_query_lanes(question)
    if not lanes or not path.exists():
        return []

    sql = (
        # The document title is repeated for every chunk, so it receives only
        # a small boost. Section names remain valuable for technical queries.
        "SELECT chunk_id, bm25(chunks_fts, 0.0, 0.0, 0.15, 2.0, 1.0) AS score "
        "FROM chunks_fts WHERE chunks_fts MATCH ?"
    )
    if source_id:
        sql += " AND source_id = ?"
    sql += " ORDER BY score LIMIT ?"

    fused: dict[str, float] = {}
    with closing(connect(path)) as connection:
        for match_query, lane_weight in lanes:
            parameters: list[Any] = [match_query]
            if source_id:
                parameters.append(source_id)
            parameters.append(max(limit * 2, limit))
            rows = connection.execute(sql, parameters).fetchall()
            for rank, row in enumerate(rows, start=1):
                chunk_id = str(row["chunk_id"])
                fused[chunk_id] = fused.get(chunk_id, 0.0) + (
                    lane_weight / (LEXICAL_RRF_K + rank)
                )

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with closing(connect(path)) as connection:
        rows = connection.execute("SELECT key, value FROM index_state").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def count_chunks(path: Path) -> int:
    if not path.exists():
        return 0
    with closing(connect(path)) as connection:
        row = connection.execute("SELECT count(*) AS count FROM chunks_fts").fetchone()
    return int(row["count"] if row else 0)


def set_state(path: Path, values: dict[str, str]) -> None:
    with closing(connect(path)) as connection, connection:
        connection.executemany(
            """
            INSERT INTO index_state(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            list(values.items()),
        )


def rebuild_from_collection(
    path: Path,
    collection: Any,
    fingerprint: str,
) -> int:
    result = collection.get(include=["documents", "metadatas"])
    ids = list(result.get("ids") or [])
    documents = list(result.get("documents") or [])
    metadatas = list(result.get("metadatas") or [])

    rows = []
    for chunk_id, content, metadata in zip(
        ids,
        documents,
        metadatas,
        strict=True,
    ):
        metadata = metadata or {}
        rows.append(
            (
                str(chunk_id),
                str(metadata.get("source_id", "")),
                str(metadata.get("document_title", "")),
                str(metadata.get("section_path", "")),
                str(content or ""),
            )
        )

    with closing(connect(path)) as connection, connection:
        connection.execute("DELETE FROM chunks_fts")
        connection.executemany(
            """
            INSERT INTO chunks_fts(chunk_id, source_id, title, section, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.execute("DELETE FROM index_state")
        connection.executemany(
            "INSERT INTO index_state(key, value) VALUES (?, ?)",
            [
                ("embedding_fingerprint", fingerprint),
                ("chunk_count", str(len(rows))),
                ("ids_fingerprint", fingerprint_ids(ids)),
            ],
        )
    return len(rows)
