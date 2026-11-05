"""Email intelligence: summary, action items, urgency, tone, reply suggestion + aggregate insights."""
import json
from collections import Counter
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from .db import conn
from .llm import get_llm

URGENCY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class ActionItem(BaseModel):
    task: str = Field(description="Concrete task, starting with a verb")
    owner: str | None = Field(None, description="Who must do it; 'me' if addressed to the recipient")
    due_date: str | None = Field(None, description="ISO date YYYY-MM-DD if stated or inferable")
    priority: Literal["high", "medium", "low"] = "medium"


class EmailAnalysis(BaseModel):
    summary: str = Field(description="2-3 sentence summary")
    action_items: list[ActionItem] = []
    urgency: Literal["critical", "high", "medium", "low"]
    urgency_reason: str
    category: Literal[
        "request", "approval", "meeting", "fyi", "complaint", "escalation", "invoice_finance", "hr", "other"
    ]
    tone: Literal["positive", "neutral", "urgent", "frustrated", "formal", "informal"]
    requires_reply: bool
    suggested_reply: str | None = Field(None, description="Short professional draft reply if a reply is needed")


PROMPT = """You are an executive assistant analysing an email for the recipient.
Today's date is {today}. Resolve relative dates ("by Friday") to ISO dates.

Urgency rubric:
- critical: outage, legal/security/compliance risk, or deadline within 24h with real consequences
- high: deadline within ~3 days, a blocked colleague, or an explicit escalation
- medium: needs action this week
- low: FYI, newsletters, no action needed

Only extract action items the email actually asks for. Do not invent facts.

From: {sender}
To: {recipients}
Subject: {subject}

{body}
"""


def analyze_email(email: dict) -> EmailAnalysis:
    llm = get_llm().with_structured_output(EmailAnalysis)
    return llm.invoke(
        PROMPT.format(
            today=date.today().isoformat(),
            sender=email.get("sender", ""),
            recipients=email.get("recipients", ""),
            subject=email.get("subject", ""),
            body=(email.get("body") or "")[:12000],
        )
    )


def analyze_and_store(email_id: int) -> dict:
    with conn() as c:
        row = c.execute("SELECT * FROM emails WHERE id=?", (email_id,)).fetchone()
        if not row:
            raise KeyError(email_id)
        result = analyze_email(dict(row))
        c.execute(
            "UPDATE emails SET analysis_json=?, analyzed_at=CURRENT_TIMESTAMP WHERE id=?",
            (result.model_dump_json(), email_id),
        )
    return result.model_dump()


def digest(limit: int = 20) -> list[dict]:
    """Analysed emails ordered by urgency, then recency."""
    with conn() as c:
        rows = c.execute("SELECT * FROM emails WHERE analysis_json IS NOT NULL").fetchall()
    items = []
    for r in rows:
        a = json.loads(r["analysis_json"])
        items.append({"id": r["id"], "sender": r["sender"], "subject": r["subject"],
                      "received_at": r["received_at"], **a})
    items.sort(key=lambda x: (URGENCY_RANK[x["urgency"]], x["received_at"]), reverse=False)
    return items[:limit]


def insights() -> dict:
    """Aggregate communication analytics across analysed emails."""
    with conn() as c:
        total = c.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
        rows = c.execute("SELECT sender, analysis_json FROM emails WHERE analysis_json IS NOT NULL").fetchall()
    urg, cat, tone, senders = Counter(), Counter(), Counter(), Counter()
    pending_replies, open_actions, urgent_by_sender = 0, 0, Counter()
    for r in rows:
        a = json.loads(r["analysis_json"])
        urg[a["urgency"]] += 1
        cat[a["category"]] += 1
        tone[a["tone"]] += 1
        senders[r["sender"]] += 1
        pending_replies += int(a["requires_reply"])
        open_actions += len(a["action_items"])
        if a["urgency"] in ("critical", "high"):
            urgent_by_sender[r["sender"]] += 1
    return {
        "total_emails": total,
        "analyzed": len(rows),
        "by_urgency": dict(urg),
        "by_category": dict(cat),
        "by_tone": dict(tone),
        "pending_replies": pending_replies,
        "open_action_items": open_actions,
        "top_senders": senders.most_common(5),
        "top_urgent_senders": urgent_by_sender.most_common(5),
    }
