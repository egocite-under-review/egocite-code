"""
Prompt loader — the only place code reads prompt text.

Every prompt lives in ``prompt/<group>/<name>.txt`` and is referenced by its
path-like id, e.g. ``load("memory/action_extract")``. No prompt string is ever
written inline in the code.

Placeholders use shell/Template syntax (``$name`` or ``${name}``) rather than
``{}`` because most prompts end with a literal JSON schema full of braces:

    render("memory/action_verify", word_min=8, word_max=20)

Substitution is "safe": an unknown ``$foo`` is left as-is instead of raising.
"""

import os
from functools import lru_cache
from string import Template

from config import PROMPT_DIR


@lru_cache(maxsize=None)
def load(name: str, strip: bool = True) -> str:
    """Return the text of ``prompt/<name>.txt``.

    ``strip`` (the default) drops trailing newlines, which is what you want for
    a standalone prompt. Pass ``strip=False`` for a fragment that gets spliced
    into a larger prompt and whose trailing newline is load-bearing.
    """
    path = os.path.join(PROMPT_DIR, *name.split("/")) + ".txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompt not found: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text.rstrip("\n") if strip else text


def render(name: str, strip: bool = True, **values) -> str:
    """Load ``name`` and substitute ``$placeholders`` with ``values``."""
    return Template(load(name, strip)).safe_substitute(**values)


def path(name: str) -> str:
    """Absolute path of a prompt file (for logging / provenance records)."""
    return os.path.join(PROMPT_DIR, *name.split("/")) + ".txt"
