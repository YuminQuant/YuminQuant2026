from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    source: str

    def __str__(self) -> str:
        return self.source
