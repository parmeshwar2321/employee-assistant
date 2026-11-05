"""Load documents -> chunk -> BGE embed -> Qdrant (dense) + SQLite (BM25 source of truth)."""
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct

from ..config import settings
from ..db import conn
from .retriever import get_retriever

SUPPORTED = {".txt", ".md", ".pdf"}


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_file(path: Path) -> int:
    text = load_text(path).strip()
    if not text:
        return 0
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    pieces = splitter.split_text(text)
    source = path.name
    r = get_retriever()

    # idempotent re-ingest: drop old chunks of this source
    r.delete_source(source)

    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}")) for i in range(len(pieces))]
    vectors = r.embed_docs(pieces)
    r.qdrant.upsert(
        collection_name=settings.collection,
        points=[
            PointStruct(id=ids[i], vector=vectors[i], payload={"source": source, "chunk_index": i})
            for i in range(len(pieces))
        ],
    )
    with conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO chunks(id, source, chunk_index, text) VALUES (?,?,?,?)",
            [(ids[i], source, i, pieces[i]) for i in range(len(pieces))],
        )
    r.refresh_bm25()
    return len(pieces)


def ingest_dir(directory: Path) -> dict[str, int]:
    return {
        p.name: ingest_file(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED
    }
