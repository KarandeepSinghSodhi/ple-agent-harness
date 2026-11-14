"""Living Engineering Ledger for long-horizon task state retention.

Prevents context amnesia during multi-hour runs by storing verified
invariants, applicable building codes, deliverable targets, and PE review
redlines in structured state outside the append-only chat history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class EngineeringLedger:
    discipline: str = "Unknown"
    applicable_codes: list[str] = field(default_factory=list)
    invariants: dict[str, str] = field(default_factory=dict)
    deliverables: dict[str, str] = field(default_factory=dict)
    redlines: list[str] = field(default_factory=list)

    def update(
        self,
        discipline: str | None = None,
        add_codes: list[str] | None = None,
        set_invariants: dict[str, str] | None = None,
        set_deliverables: dict[str, str] | None = None,
    ) -> str:
        if discipline:
            self.discipline = discipline
        if add_codes:
            for code in add_codes:
                if code not in self.applicable_codes:
                    self.applicable_codes.append(code)
        if set_invariants:
            self.invariants.update(set_invariants)
        if set_deliverables:
            self.deliverables.update(set_deliverables)
        return self.render_summary()

    def render_summary(self) -> str:
        lines = [
            "=== LIVING ENGINEERING LEDGER ===",
            f"Discipline: {self.discipline}",
            f"Applicable Codes: {', '.join(self.applicable_codes) if self.applicable_codes else 'None recorded yet'}",
            "Verified Invariants:",
        ]
        if self.invariants:
            for k, v in self.invariants.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append("  (none yet)")
        lines.append("Deliverables Status:")
        if self.deliverables:
            for path, status in self.deliverables.items():
                lines.append(f"  - {path}: {status}")
        else:
            lines.append("  (none yet)")
        if self.redlines:
            lines.append("Open PE Redlines:")
            for r in self.redlines:
                lines.append(f"  [!] {r}")
        lines.append("=================================")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "discipline": self.discipline,
            "applicable_codes": self.applicable_codes,
            "invariants": self.invariants,
            "deliverables": self.deliverables,
            "redlines": self.redlines,
        }, indent=2)
