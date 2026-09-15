"""Small display-only Markdown normalizations for the chat UI."""

from __future__ import annotations

import re

_BRANCH_LINE_BREAK = re.compile(r"(?<!\n)\n(?=•\s)")


def prepare_chat_markdown(text: str) -> str:
    """Make organisation branches visibly separate in Streamlit Markdown.

    The agent already returns a logical line break before each ``•`` branch.
    CommonMark renders a lone newline as a space, though. Promote only these
    branch boundaries to Markdown hard breaks; paragraphs and all other model
    text stay untouched.
    """

    return _BRANCH_LINE_BREAK.sub("  \n", text)
