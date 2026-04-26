"""
Coerce TRL/GRPO `completion` values to a single string for `parse_action`.

Chat-style data often yields a list of message parts or content blocks, not a
decoded string (unlike `evaluate()` which uses tokenizer.decode). This module
has no heavy dependencies so it can be unit-tested without Unsloth/CUDA.
"""

from __future__ import annotations

import json
from typing import Any


def _completion_to_text(completion: Any) -> str:
    """
    Turn TRL/Unsloth completion into one string, mirroring a single decode path.

    Handles: str, list of str, list[dict] (e.g. chat parts with "content" or
    OpenAI-style {"type": "text", "text": "..."}), dict with "content" or
    "text", and nested list content (multimodal).
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, bytes):
        return completion.decode("utf-8", errors="replace")
    if isinstance(completion, dict):
        if "content" in completion and completion["content"] is not None:
            return _completion_to_text(completion["content"])
        if "text" in completion:
            return str(completion["text"])
        return json.dumps(completion, ensure_ascii=False)
    if isinstance(completion, (list, tuple)):
        parts: list[str] = []
        for item in completion:
            chunk = _completion_to_text(item)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts)
    return str(completion)
