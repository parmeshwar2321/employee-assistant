import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import email_intel
from .config import settings
from .db import conn, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    from .rag.graph import build_graph
    from .rag.retriever import get_retriever

    get_retriever()  # warm embedder + indexes
    app.state.graph = build_graph()
    yield


app = FastAPI(title="Employee AI Assistant", version="0.1.0", lifespan=lifespan)


# ---------- schemas ----------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class EmailIn(BaseModel):
    sender: str
    subject: str
    body: str
    recipients: str = ""
    received_at: str | None = None


# ---------- health ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": settings.llm_model}


# ---------- documents ----------
@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    from .rag.ingest import SUPPORTED, ingest_file

    from pathlib import Path

    name = Path(file.filename or "upload.txt").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Supported types: {sorted(SUPPORTED)}")
    docs_dir = settings.data_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    dest = docs_dir / name
    dest.write_bytes(file.file.read())
    return {"source": name, "chunks": ingest_file(dest)}


@app.post("/documents/ingest-dir")
def ingest_directory(path: str = "sample_docs"):
    from pathlib import Path

    from .rag.ingest import ingest_dir

    p = Path(path)
    if not p.is_dir():
        raise HTTPException(404, "Directory not found")
    return ingest_dir(p)


@app.get("/documents")
def list_documents():
    with conn() as c:
        rows = c.execute("SELECT source, COUNT(*) AS chunks FROM chunks GROUP BY source").fetchall()
    return [dict(r) for r in rows]


# ---------- chat (RAG) ----------
@app.post("/chat")
def chat(req: ChatRequest):
    from .rag.graph import initial_input

    thread_id = req.thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        out = app.state.graph.invoke(initial_input(req.message), cfg)
    except Exception as e:  # surface LLM/key errors cleanly
        raise HTTPException(502, f"Pipeline error: {e}")
    with conn() as c:
        c.execute(
            "INSERT INTO threads(thread_id, title) VALUES (?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET updated_at=CURRENT_TIMESTAMP",
            (thread_id, req.message[:60]),
        )
    sources, seen = [], set()
    for i, d in enumerate(out.get("docs", []), 1):
        sources.append({"ref": i, "source": d["source"], "chunk_index": d["chunk_index"],
                        "snippet": d["text"][:240]})
    return {"thread_id": thread_id, "answer": out["answer"], "grounded": out.get("grounded", True),
            "sources": sources, "retrieval_rounds": out.get("rounds", 0), "trace": out.get("trace", [])}


@app.get("/threads")
def list_threads():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM threads ORDER BY updated_at DESC")]


@app.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    state = app.state.graph.get_state({"configurable": {"thread_id": thread_id}})
    msgs = state.values.get("messages") if state and state.values else None
    if not msgs:
        raise HTTPException(404, "Thread not found")
    return {"thread_id": thread_id, "messages": [{"role": m.type, "content": m.content} for m in msgs]}


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    cp = app.state.graph.checkpointer
    cp.delete_thread(thread_id)
    with conn() as c:
        c.execute("DELETE FROM threads WHERE thread_id=?", (thread_id,))
    return {"deleted": thread_id}


# ---------- email intelligence ----------
@app.post("/emails")
def add_email(e: EmailIn, analyze: bool = False):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO emails(sender, recipients, subject, body, received_at) "
            "VALUES (?,?,?,?,COALESCE(?, CURRENT_TIMESTAMP))",
            (e.sender, e.recipients, e.subject, e.body, e.received_at),
        )
        email_id = cur.lastrowid
    result = email_intel.analyze_and_store(email_id) if analyze else None
    return {"id": email_id, "analysis": result}


@app.post("/emails/analyze-text")
def analyze_text(e: EmailIn):
    """Stateless analysis, nothing is stored."""
    return email_intel.analyze_email(e.model_dump()).model_dump()


@app.post("/emails/{email_id}/analyze")
def analyze_one(email_id: int):
    try:
        return email_intel.analyze_and_store(email_id)
    except KeyError:
        raise HTTPException(404, "Email not found")


@app.post("/emails/analyze-pending")
def analyze_pending():
    with conn() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM emails WHERE analysis_json IS NULL")]
    done = [i for i in ids if email_intel.analyze_and_store(i)]
    return {"analyzed": done}


@app.get("/emails")
def list_emails():
    with conn() as c:
        rows = c.execute("SELECT id, sender, subject, received_at, analysis_json FROM emails ORDER BY id DESC").fetchall()
    return [{**{k: r[k] for k in ("id", "sender", "subject", "received_at")},
             "analysis": json.loads(r["analysis_json"]) if r["analysis_json"] else None} for r in rows]


@app.get("/emails/digest")
def email_digest(limit: int = 20):
    return email_intel.digest(limit)


@app.get("/emails/insights")
def email_insights():
    return email_intel.insights()


@app.post("/emails/sync")
def sync_inbox(limit: int = 20, analyze: bool = True):
    """Pull the latest emails over IMAP (skips ones already stored), optionally analyse them."""
    from . import email_sync

    try:
        new_ids = email_sync.sync_imap(limit)
    except Exception as e:
        raise HTTPException(400, f"IMAP sync failed: {e}")
    if analyze:
        for i in new_ids:
            email_intel.analyze_and_store(i)
    return {"new_emails": len(new_ids), "ids": new_ids, "analyzed": analyze}


@app.post("/emails/upload-eml")
def upload_eml(file: UploadFile = File(...), analyze: bool = True):
    """No credentials needed: upload a downloaded .eml file."""
    from . import email_sync

    eid = email_sync.store_email(email_sync.parse_message(file.file.read()))
    if eid is None:
        return {"id": None, "note": "duplicate, already stored"}
    return {"id": eid, "analysis": email_intel.analyze_and_store(eid) if analyze else None}
