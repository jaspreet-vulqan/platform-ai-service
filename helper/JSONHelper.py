"""Tolerant JSON extraction from LLM output.

Local models sometimes wrap JSON in ```json fences or add a sentence of prose.
This pulls out the first JSON object regardless.
"""

import json
from typing import Any


def extractJSON(text: str) -> Any:
    """Parse a JSON object from possibly-noisy LLM text. Raises ValueError."""
    s = (text or "").strip()

    # Strip a leading/trailing markdown code fence if present.
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s[:4].lower() == "json":
            s = s[4:].strip()

    # Fast path: it's already clean JSON.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fallback: slice from the first '{' to the last '}'.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(s[start : end + 1])
