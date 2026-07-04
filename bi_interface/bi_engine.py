"""Business-Intelligence layer: natural-language questions over the shelf inventory (Module 5).

Two answer paths:

1. Rule-based (always available): pattern-matches the question to inventory intents (counts,
   top categories, empty space, per-category lookups, review items, cross-scan trends) and
   computes the answer directly from the pandas frames. Deterministic and fast.

2. LLM path (optional): if an Ollama server is reachable, the question plus a compact JSON
   summary of the current inventory is sent to a local model for a free-form answer. This is the
   Module-5 "LangGraph/Ollama NL interface" hook — enabled automatically when Ollama is up and
   silently skipped otherwise.

`answer()` returns an `Answer` with the text plus optional structured data the UI can chart.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

SUGGESTED_QUESTIONS = [
    "How many products were detected?",
    "What are the top 5 categories?",
    "How much empty shelf space is there?",
    "Which items need manual review?",
    "Is this shelf mixed or category-specific?",
    "How many soft drinks are on the shelf?",
    "What is the category breakdown?",
    "How has the total product count changed over time?",
]


@dataclass
class Answer:
    text: str
    source: str = "rule-based"  # or "llm:<model>"
    table: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)


# -- aggregation helpers ---------------------------------------------------
def _known(items: pd.DataFrame) -> pd.DataFrame:
    if items is None or items.empty:
        return items
    return items[items["category"].str.lower() != "unknown"]


def category_counts(items: pd.DataFrame) -> pd.Series:
    k = _known(items)
    if k is None or k.empty:
        return pd.Series(dtype=int)
    return k["category"].value_counts()


def subcategory_counts(items: pd.DataFrame) -> pd.Series:
    k = _known(items)
    if k is None or k.empty:
        return pd.Series(dtype=int)
    sub = k[k["subcategory"].str.lower() != "unknown"]
    return sub["subcategory"].value_counts() if not sub.empty else pd.Series(dtype=int)


def inventory_summary(items: pd.DataFrame, scans: pd.DataFrame | None = None) -> dict:
    counts = category_counts(items)
    total = int(len(items)) if items is not None else 0
    unknown = int((items["category"].str.lower() == "unknown").sum()) if total else 0
    return {
        "total_items": total,
        "distinct_categories": int(counts.size),
        "unknown_items": unknown,
        "top_categories": counts.head(10).to_dict(),
        "num_scans": int(len(scans)) if scans is not None else None,
    }


# -- rule-based intent matching --------------------------------------------
def _answer_rule_based(question: str, items: pd.DataFrame, scans: pd.DataFrame | None) -> Answer:
    q = (question or "").lower().strip()
    counts = category_counts(items)
    total = int(len(items)) if items is not None else 0

    if total == 0:
        return Answer("No inventory yet — analyze a shelf image first.")

    if "empty" in q or "space" in q or "gap" in q or "stock" in q:
        if scans is not None and not scans.empty:
            pct = float(scans.iloc[0]["empty_pct"]) * 100
            band = "High" if pct >= 55 else "Moderate" if pct >= 25 else "Low"
            return Answer(f"The most recent shelf shows about **{pct:.0f}% empty space** ({band}). "
                          f"Higher empty space suggests restocking may be needed.")
        return Answer("Empty-space data is per scan; analyze an image first.")

    if "review" in q or "unknown" in q or "unclear" in q or "manual" in q:
        n = int((items["category"].str.lower() == "unknown").sum())
        return Answer(f"**{n}** item{'s' if n != 1 else ''} could not be confidently identified "
                      f"and should be reviewed manually."
                      if n else "No items require manual review.")

    if "mixed" in q or "specific" in q or ("shelf" in q and "type" in q):
        d = int(counts.size)
        kind = "mixed" if d > 1 else "category-specific" if d == 1 else "empty"
        return Answer(f"This inventory looks **{kind}** — it spans {d} distinct categories.")

    if any(w in q for w in ["over time", "trend", "changed", "history", "compare scans"]):
        if scans is None or len(scans) < 2:
            return Answer("Need at least two scans to show a trend. Analyze more images.")
        s = scans.sort_values("id")
        first, last = int(s.iloc[0]["num_items"]), int(s.iloc[-1]["num_items"])
        delta = last - first
        arrow = "increased" if delta > 0 else "decreased" if delta < 0 else "stayed flat"
        return Answer(f"Across **{len(s)} scans**, total detected products {arrow} "
                      f"from {first} to {last} ({'+' if delta >= 0 else ''}{delta}).",
                      table=s[["id", "ts", "num_items", "distinct_categories", "empty_pct"]])

    if "top" in q or "most" in q or ("categor" in q and ("what" in q or "which" in q)):
        m = re.search(r"top\s+(\d+)", q)
        n = int(m.group(1)) if m else 5
        top = counts.head(n)
        lines = ", ".join(f"{c} ({v})" for c, v in top.items())
        return Answer(f"Top {len(top)} categor{'ies' if len(top) != 1 else 'y'}: {lines}.",
                      table=top.rename_axis("category").reset_index(name="count"))

    if "breakdown" in q or "distribution" in q or "all categor" in q or "composition" in q:
        return Answer(f"Detected {total} products across {counts.size} categories. See the table.",
                      table=counts.rename_axis("category").reset_index(name="count"))

    if "how many" in q or "count" in q or "number of" in q:
        if re.search(r"how many (products|items|things|skus)\b", q) or "total" in q:
            return Answer(f"**{total}** products were detected in the current inventory.")
        matches = _match_categories(q, counts.index.tolist())
        if matches:
            tot = int(counts[matches].sum())
            detail = ", ".join(f"{c} ({int(counts[c])})" for c in matches)
            return Answer(f"Found **{tot}** matching product{'s' if tot != 1 else ''}: {detail}.",
                          table=counts[matches].rename_axis("category").reset_index(name="count"))
        return Answer(f"I couldn't match that to a known category. The inventory has {total} "
                      f"products across {counts.size} categories.")

    top = counts.head(5)
    lines = ", ".join(f"{c} ({v})" for c, v in top.items())
    return Answer(f"The inventory has **{total} products** across **{counts.size} categories**. "
                  f"Top categories: {lines}. Ask about empty space, review items, or a category.",
                  table=counts.head(10).rename_axis("category").reset_index(name="count"))


def _match_categories(question: str, categories: list[str]) -> list[str]:
    q_words = set(re.findall(r"[a-z]+", question.lower()))
    stop = {"how", "many", "much", "the", "are", "there", "on", "shelf", "count", "of", "number",
            "products", "items", "a", "an", "is", "do", "we", "have", "and", "in", "this"}
    q_words -= stop
    return [c for c in categories if q_words & set(re.findall(r"[a-z]+", c.lower()))]


# -- optional Ollama LLM path ----------------------------------------------
def ollama_available(timeout: float = 1.5) -> bool:
    try:
        import requests

        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def _answer_llm(question: str, items: pd.DataFrame, scans: pd.DataFrame | None) -> Answer | None:
    try:
        import requests

        summary = inventory_summary(items, scans)
        prompt = (
            "You are a retail shelf analytics assistant. Answer the user's question using ONLY "
            "the inventory summary JSON below. Be concise and specific with numbers.\n\n"
            f"INVENTORY SUMMARY:\n{json.dumps(summary, indent=2)}\n\nQUESTION: {question}\n\nANSWER:"
        )
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=60)
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        return Answer(text, source=f"llm:{OLLAMA_MODEL}", meta={"summary": summary}) if text else None
    except Exception:
        return None


def answer(question: str, items: pd.DataFrame, scans: pd.DataFrame | None = None,
           use_llm: bool = True) -> Answer:
    """Answer a natural-language inventory question. Prefers Ollama when reachable."""
    if use_llm and ollama_available():
        llm = _answer_llm(question, items, scans)
        if llm is not None:
            return llm
    return _answer_rule_based(question, items, scans)
