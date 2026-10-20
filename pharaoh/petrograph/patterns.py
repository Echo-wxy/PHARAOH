"""Well-pattern templates in units of the pattern spacing.

Each template lists (dx, dy, role) offsets relative to the pattern anchor,
with role "I" for an injector and "P" for a producer.
"""
from __future__ import annotations

PATTERN_TEMPLATES: dict[str, list[tuple[float, float, str]]] = {
    "five_spot": [(0, 0, "I"), (-1, -1, "P"), (1, -1, "P"), (-1, 1, "P"), (1, 1, "P")],
    "inverted_five_spot": [(0, 0, "P"), (-1, -1, "I"), (1, -1, "I"), (-1, 1, "I"), (1, 1, "I")],
    "nine_spot": [(0, 0, "I")] + [(dx, dy, "P") for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)],
    "inverted_nine_spot": [(0, 0, "P")] + [(dx, dy, "I") for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)],
    "line_drive": [(x, -0.5, "I") for x in (-1, 0, 1)] + [(x, 0.5, "P") for x in (-1, 0, 1)],
    "staggered_line_drive": [(x, -0.5, "I") for x in (-1, 0, 1)] + [(x, 0.5, "P") for x in (-0.5, 0.5)],
    "peripheral": [(1.5 * dx, 1.5 * dy, "I") for dx, dy in ((1, 0), (0.5, 0.87), (-0.5, 0.87), (-1, 0), (-0.5, -0.87), (0.5, -0.87))]
    + [(-0.4, 0, "P"), (0.4, 0, "P")],
}

PATTERN_LABELS: dict[str, str] = {
    "five_spot": "five-spot",
    "inverted_five_spot": "inverted five-spot",
    "nine_spot": "nine-spot",
    "inverted_nine_spot": "inverted nine-spot",
    "line_drive": "direct line drive",
    "staggered_line_drive": "staggered line drive",
    "peripheral": "peripheral flood",
}


def rotate(offsets: list[tuple[float, float, str]], quarter_turns: int) -> list[tuple[float, float, str]]:
    """Rotate template offsets by multiples of 90 degrees."""
    out = []
    for dx, dy, role in offsets:
        for _ in range(quarter_turns % 4):
            dx, dy = -dy, dx
        out.append((dx, dy, role))
    return out
