"""Privacy: what the assistant and the API are allowed to see.

Two mechanisms, both structural rather than by convention:

1. **A separate registry for sensitive customer flags.** Whether a home has
   life-support equipment, a threat marker or a protected-contact note lives
   in ``SensitiveRegistry``, which is never passed to the risk model, the
   restoration simulation, or any language model. The only thing that
   leaves it is a yes/no "eligible for proactive alerts" answer, computed
   here.

2. **Redaction at the boundary.** ``redact`` strips internal asset ids and
   raw coordinates from any payload before it reaches a customer-facing
   surface, and an ``AuditLog`` records every tool call: which tool, a hash
   of the input, which fields went out, which were removed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

_ID_RE = re.compile(r"^F\d+-")  # internal asset ids look like F3-2-P4
SENSITIVE_KEYS = {"life_support", "threat", "protected_contact", "medical"}
INTERNAL_KEYS = {"x", "y", "px", "py", "asset_id", "transformer", "parent"}


@dataclass
class SensitiveRegistry:
    """Per-premise flags that must never reach a model."""

    _flags: dict[str, dict[str, bool]] = field(default_factory=dict)

    def set(self, premise_id: str, **flags: bool) -> None:
        self._flags.setdefault(premise_id, {}).update(flags)

    def alert_eligible(self, premise_id: str) -> bool:
        """The one derived fact that may leave the registry."""
        f = self._flags.get(premise_id, {})
        return bool(f.get("life_support") or f.get("medical"))

    def __contains__(self, premise_id: str) -> bool:
        return premise_id in self._flags


def redact(payload: Any, *, customer_facing: bool = True) -> tuple[Any, list[str]]:
    """Return a copy with internal fields removed, plus the list of removed keys."""
    removed: list[str] = []

    def walk(v: Any, path: str) -> Any:
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if k in SENSITIVE_KEYS or (customer_facing and k in INTERNAL_KEYS):
                    removed.append(f"{path}.{k}" if path else k)
                    continue
                if isinstance(x, str) and _ID_RE.match(x) and customer_facing:
                    removed.append(f"{path}.{k}" if path else k)
                    continue
                out[k] = walk(x, f"{path}.{k}" if path else k)
            return out
        if isinstance(v, list):
            return [walk(x, f"{path}[]") for x in v]
        return v

    return walk(payload, ""), sorted(set(removed))


@dataclass
class AuditEntry:
    ts: float
    tool: str
    input_hash: str
    fields_out: list[str]
    fields_removed: list[str]
    audience: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class AuditLog:
    def __init__(self, keep: int = 500):
        self._entries: deque[AuditEntry] = deque(maxlen=keep)

    def record(self, tool: str, inputs: dict, output: dict, removed: list[str], audience: str) -> None:
        h = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self._entries.append(AuditEntry(time.time(), tool, h, sorted(output.keys()), removed, audience))

    def recent(self, n: int = 50) -> list[dict]:
        return [e.to_dict() for e in list(self._entries)[-n:]]

    def __len__(self) -> int:
        return len(self._entries)
