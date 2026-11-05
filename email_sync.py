"""Free email ingestion: IMAP sync (stdlib only) and .eml parsing. No OAuth or paid APIs."""
import email
import hashlib
import html
import imaplib
import re
from email import policy
from email.utils import parsedate_to_datetime

from .config import settings
from .db import conn


def _clean(text: str, is_html: bool) -> str:
    if is_html:
        text = re.sub(r"(?is)<(script|style).*?</\1>", "", text)
        text = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", text)
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^On .+wrote:$", s) or s.startswith("-----Original Message"):
            break  # drop quoted reply chain
        if s.startswith(">"):
            continue
        lines.append(ln.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_message(raw: bytes) -> dict:
    msg = email.message_from_bytes(raw, policy=policy.default)
    part = msg.get_body(preferencelist=("plain", "html"))
    body = _clean(part.get_content(), part.get_content_type() == "text/html") if part else ""
    try:
        received = parsedate_to_datetime(msg["Date"]).isoformat(sep=" ", timespec="seconds")
    except Exception:
        received = None
    sender, subject = str(msg["From"] or ""), str(msg["Subject"] or "(no subject)")
    mid = str(msg["Message-ID"] or "") or hashlib.sha1(f"{sender}{subject}{received}".encode()).hexdigest()
    return {"message_id": mid, "sender": sender, "recipients": str(msg["To"] or ""),
            "subject": subject, "body": body, "received_at": received}


def store_email(d: dict) -> int | None:
    """Insert unless this Message-ID is already stored. Returns new id or None if duplicate."""
    with conn() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO emails(message_id, sender, recipients, subject, body, received_at) "
            "VALUES (?,?,?,?,?,COALESCE(?, CURRENT_TIMESTAMP))",
            (d["message_id"], d["sender"], d["recipients"], d["subject"], d["body"], d["received_at"]),
        )
        return cur.lastrowid if cur.rowcount else None


def sync_imap(limit: int = 20) -> list[int]:
    if not (settings.imap_user and settings.imap_password):
        raise RuntimeError("Set IMAP_USER and IMAP_PASSWORD in .env")
    new_ids: list[int] = []
    box = imaplib.IMAP4_SSL(settings.imap_host)
    try:
        box.login(settings.imap_user, settings.imap_password)
        box.select("INBOX", readonly=True)  # read-only: never marks mail as read
        _, data = box.search(None, "ALL")
        for num in data[0].split()[-limit:]:
            _, parts = box.fetch(num, "(BODY.PEEK[])")
            raw = next(p[1] for p in parts if isinstance(p, tuple))
            eid = store_email(parse_message(raw))
            if eid:
                new_ids.append(eid)
    finally:
        try:
            box.logout()
        except Exception:
            pass
    return new_ids
