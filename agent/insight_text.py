"""Small output guards shared by evidence-bounded AI interpretations."""

from __future__ import annotations

import re


_META_QUESTION_CLAUSE = re.compile(
    r"[，；;]?\s*"
    r"(?:(?:该|此)图|图\s*\d+\s*)?"
    r"(?:主要\s*)?可(?:用于\s*)?回答(?:以下)?"
    r"(?:[“\"][^”\"\n]+[”\"]|[^。！？\n]+)"
    r"(?:这一问题|这个问题|问题)?"
)


def remove_meta_question_phrasing(text: str) -> str:
    """Remove chart-guide boilerplate such as ``可回答“……”``.

    The chart description before the clause is retained, while meta commentary
    about what question a figure can answer is removed from both regular and
    benchmark LLM responses.
    """
    cleaned = _META_QUESTION_CLAUSE.sub("", str(text))
    cleaned = re.sub(r"[，；;]\s*([。！？])", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s*[，；;。]\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
