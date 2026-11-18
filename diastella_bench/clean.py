"""How the hosted tool for coding agents cleans a piece of text before using or returning it."""
import re
import unicodedata


def quoted(text: str | None, limit: int = 140) -> str:
    """A stranger's text made safe to hand to an agent: one line, no markup, no links, no control characters, short."""
    # no control characters and nothing a reader cannot see (zero-width and direction marks, private-use and tag
    # characters): hidden text is the usual carrier of instructions meant for a model
    text = "".join(c for c in unicodedata.normalize("NFKC", text or "") if c >= " " and unicodedata.category(c) not in ("Cf", "Co", "Cs", "Cn"))
    text = re.sub(r"https?://\S+", "[link]", text)
    text = re.sub(r"[`<>{}\[\]|\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]
