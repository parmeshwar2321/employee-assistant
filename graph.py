"""Self-corrective RAG as a LangGraph state machine.

contextualize -> retrieve -> grade
  grade: insufficient and budget left -> rewrite -> retrieve (loop)
  grade: sufficient or budget spent   -> generate -> verify
  verify: ungrounded and retry left   -> generate (with feedback)
  verify: grounded or out of retries  -> finalize
"""
import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ..config import settings
from ..db import CHECKPOINT_DB
from ..llm import get_llm
from .retriever import get_retriever

MAX_REGENS = 1


class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    query: str
    candidates: list[dict]
    docs: list[dict]          # accepted, relevant evidence (accumulates across rounds)
    missing: str
    sufficient: bool
    rounds: int
    regens: int
    feedback: str
    answer: str
    grounded: bool
    trace: list[str]


class Grade(BaseModel):
    relevant_ids: list[int] = Field(description="Numbers of passages that help answer the question")
    sufficient: bool = Field(description="True if the relevant passages fully answer the question")
    missing: str = Field("", description="If insufficient: what information is still missing")


class Verdict(BaseModel):
    grounded: bool = Field(description="True only if every claim is supported by the passages")
    unsupported_claims: list[str] = []


class Rewrite(BaseModel):
    query: str


def _fmt(docs: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (source: {d['source']})\n{d['text']}" for i, d in enumerate(docs, 1))


def _t(state: RAGState, msg: str) -> list[str]:
    return state.get("trace", []) + [msg]


# ---------------- nodes ----------------
def contextualize(state: RAGState) -> dict:
    q = state["question"]
    history = state["messages"][:-1][-6:]
    if not history:
        return {"query": q, "trace": _t(state, "contextualize: first turn")}
    convo = "\n".join(f"{m.type}: {m.content}" for m in history)
    out = get_llm().with_structured_output(Rewrite).invoke(
        f"Rewrite the final question as a standalone search query using the conversation for context. "
        f"Keep names/terms exact.\n\nConversation:\n{convo}\n\nFinal question: {q}"
    )
    return {"query": out.query, "trace": _t(state, f"contextualize: '{out.query}'")}


def retrieve(state: RAGState) -> dict:
    rounds = state.get("rounds", 0) + 1
    cands = get_retriever().search(state["query"])
    return {"candidates": cands, "rounds": rounds,
            "trace": _t(state, f"retrieve[{rounds}]: {len(cands)} candidates for '{state['query']}'")}


def grade(state: RAGState) -> dict:
    cands = state.get("candidates", [])
    kept = {d["id"]: d for d in state.get("docs", [])}
    if not cands:
        return {"sufficient": False, "missing": "no documents found", "trace": _t(state, "grade: nothing retrieved")}
    g = get_llm().with_structured_output(Grade).invoke(
        f"Question: {state['question']}\n\nPassages:\n{_fmt(cands)}\n\n"
        "Select the passages relevant to the question and decide whether together with any previously "
        f"accepted evidence they suffice.\nPreviously accepted evidence:\n{_fmt(list(kept.values())) or 'none'}"
    )
    for n in g.relevant_ids:
        if 1 <= n <= len(cands):
            kept[cands[n - 1]["id"]] = cands[n - 1]
    docs = list(kept.values())
    return {"docs": docs, "sufficient": g.sufficient and bool(docs), "missing": g.missing,
            "trace": _t(state, f"grade: +{len(g.relevant_ids)} relevant, sufficient={g.sufficient}")}


def rewrite(state: RAGState) -> dict:
    out = get_llm().with_structured_output(Rewrite).invoke(
        f"The search query below did not surface enough evidence.\nQuestion: {state['question']}\n"
        f"Previous query: {state['query']}\nMissing: {state.get('missing', '')}\n"
        "Write a better query: use synonyms, internal jargon, or target the missing facts."
    )
    return {"query": out.query, "trace": _t(state, f"rewrite: '{out.query}'")}


def generate(state: RAGState) -> dict:
    docs = state.get("docs", [])
    if not docs:
        return {"answer": "I couldn't find anything in the internal documents that answers this.",
                "grounded": True, "trace": _t(state, "generate: no evidence")}
    fb = state.get("feedback")
    fix = f"\nA reviewer flagged unsupported claims; remove or correct them: {fb}\n" if fb else ""
    history = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][:-1][-6:])
    resp = get_llm(0.2).invoke(
        "You are an internal company assistant. Answer ONLY from the numbered passages. "
        "Cite sources inline like [1], [2]. If the passages are incomplete, say what is missing. "
        f"Be concise.{fix}\n\nConversation so far:\n{history or 'none'}\n\n"
        f"Passages:\n{_fmt(docs)}\n\nQuestion: {state['question']}"
    )
    return {"answer": resp.content if isinstance(resp.content, str) else str(resp.content),
            "regens": state.get("regens", 0) + (1 if fb else 0),
            "trace": _t(state, "generate: draft written")}


def verify(state: RAGState) -> dict:
    docs = state.get("docs", [])
    if not docs:
        return {"grounded": True}
    v = get_llm().with_structured_output(Verdict).invoke(
        f"Passages:\n{_fmt(docs)}\n\nAnswer:\n{state['answer']}\n\n"
        "Is every factual claim in the answer supported by the passages?"
    )
    return {"grounded": v.grounded, "feedback": "; ".join(v.unsupported_claims),
            "trace": _t(state, f"verify: grounded={v.grounded}")}


def finalize(state: RAGState) -> dict:
    ans = state["answer"]
    if not state.get("grounded", True):
        ans += "\n\n_Note: some statements could not be fully verified against the source documents._"
    return {"answer": ans, "messages": [AIMessage(content=ans)]}


# ---------------- routing ----------------
def after_grade(state: RAGState) -> str:
    if state.get("sufficient") or state.get("rounds", 0) >= settings.max_retrieval_rounds:
        return "generate"
    return "rewrite"


def after_verify(state: RAGState) -> str:
    if state.get("grounded") or state.get("regens", 0) >= MAX_REGENS:
        return "finalize"
    return "generate"


def build_graph():
    g = StateGraph(RAGState)
    for name, fn in [("contextualize", contextualize), ("retrieve", retrieve), ("grade", grade),
                     ("rewrite", rewrite), ("generate", generate), ("verify", verify), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"finalize": "finalize", "generate": "generate"})
    g.add_edge("finalize", END)
    saver = SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))
    return g.compile(checkpointer=saver)


def initial_input(question: str) -> dict:
    """Per-turn reset of transient fields; `messages` persists via the checkpointer."""
    return {"messages": [HumanMessage(content=question)], "question": question, "rounds": 0,
            "regens": 0, "docs": [], "candidates": [], "feedback": "", "trace": [], "grounded": False}
