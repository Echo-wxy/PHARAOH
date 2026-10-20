"""Prompt rendering and tool-call validation for the agent and LLM baselines.

Templates use `{{name}}` placeholders. Rendering is strict: a missing value
or an unused value raises, so prompt files and calling code cannot drift
apart silently.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pharaoh.petrograph.schema import EDGE_TYPES, LEVELS

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*\})\s*</tool_call>", re.DOTALL)

CONSTRAINT_FIELDS = ("npv_musd", "capex_musd", "eur_mmbbl", "cum_water_mmbbl", "water_cut",
                     "well_spacing_m", "well_depth_m", "n_wells", "region", "pattern_type")


class PromptError(ValueError):
    """Raised for template/variable mismatches."""


class ToolCallError(ValueError):
    """Raised when a model output is not a valid tool call."""


def placeholders(template: str) -> set[str]:
    """Names of all placeholders in a template."""
    return set(_PLACEHOLDER.findall(template))


def render(template: str, **values: Any) -> str:
    """Fill placeholders; every placeholder must be given and every value used."""
    need = placeholders(template)
    missing, unused = need - values.keys(), values.keys() - need
    if missing or unused:
        raise PromptError(f"missing={sorted(missing)} unused={sorted(unused)}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


class PromptSet:
    """Loads prompt files from a directory (default: `prompts/` at the repository root)."""

    def __init__(self, root: str | Path = "prompts") -> None:
        self.root = Path(root)
        self.tools = json.loads((self.root / "agent" / "tools.json").read_text(encoding="utf-8"))

    def text(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    def agent_system(self, reservoir: str, objectives: list[dict[str, Any]]) -> str:
        """System prompt with the objective table and tool schema filled in."""
        rows = "\n".join(
            f"- {o['name']}: {'maximize' if o['sense'] == 'max' else 'minimize'} ({o.get('unit', '')})" for o in objectives
        )
        schema = json.dumps(self.tools, indent=2)
        return render(self.text("agent/system.md"), reservoir=reservoir, objective_table=rows,
                      constraint_fields=", ".join(CONSTRAINT_FIELDS), tool_schema=schema)

    def agent_user(self, query: str, max_steps: int) -> str:
        return render(self.text("agent/user.md"), query=query, max_steps=max_steps)

    def observation(self, step: int, tool: str, results: list[dict[str, Any]],
                    archive: list[dict[str, Any]], remaining: int) -> str:
        """Render an observation; each node dict needs id, type, and estimates."""
        return render(self.text("agent/observation.md"), step=step, tool=tool,
                      results=format_nodes(results), n_archive=len(archive),
                      archive=format_nodes(archive), remaining=remaining)

    def format_error(self, error: str) -> str:
        return render(self.text("agent/format_error.md"), error=error)


def format_nodes(nodes: list[dict[str, Any]]) -> str:
    """One line per node: id, type, surrogate estimates, short description."""
    if not nodes:
        return "(none)"
    lines = []
    for n in nodes:
        est = ", ".join(f"{k}={v:.3g}" for k, v in n.get("estimates", {}).items())
        desc = n.get("summary", "")
        lines.append(f"- {n['id']} [{n['type']}] {est}" + (f" | {desc}" if desc else ""))
    return "\n".join(lines)


def _check_subquery(sq: Any, objectives: set[str]) -> None:
    if not isinstance(sq, dict):
        raise ToolCallError("subquery must be an object")
    for key in ("weights", "reference_point", "constraints"):
        if key not in sq:
            raise ToolCallError(f"subquery.{key} is required")
    w = sq["weights"]
    if not isinstance(w, dict) or not w:
        raise ToolCallError("subquery.weights must be a non-empty object")
    for k, v in w.items():
        if k not in objectives:
            raise ToolCallError(f"unknown objective in weights: {k}")
        if not isinstance(v, (int, float)) or v < 0:
            raise ToolCallError(f"weight for {k} must be a non-negative number")
    if sum(w.values()) <= 0:
        raise ToolCallError("weights must not all be zero")
    rp = sq["reference_point"]
    if rp != "utopia" and not (isinstance(rp, dict) and set(rp) <= objectives):
        raise ToolCallError('reference_point must be "utopia" or an objective -> value object')
    if not isinstance(sq["constraints"], list):
        raise ToolCallError("subquery.constraints must be a list")
    for c in sq["constraints"]:
        if not isinstance(c, dict) or set(c) != {"field", "op", "value"}:
            raise ToolCallError("each constraint needs exactly field, op, value")
        if c["field"] not in CONSTRAINT_FIELDS:
            raise ToolCallError(f"unknown constraint field: {c['field']}")
        if c["op"] not in (">=", "<=", "=="):
            raise ToolCallError(f"unknown operator: {c['op']}")


def parse_tool_call(text: str, objectives: list[str], max_k: int = 50) -> dict[str, Any]:
    """Extract and validate the single tool call in a model turn."""
    found = _TOOL_CALL.findall(text)
    if len(found) != 1:
        raise ToolCallError(f"expected exactly one <tool_call>, found {len(found)}")
    try:
        call = json.loads(found[0])
    except json.JSONDecodeError as e:
        raise ToolCallError(f"tool call is not valid JSON: {e.msg}") from None
    name, args = call.get("name"), call.get("arguments", {})
    obj = set(objectives)
    if name == "finish":
        return {"name": "finish", "arguments": args}
    if name not in ("ProxySearch", "ProxyExpand"):
        raise ToolCallError(f"unknown tool: {name}")
    _check_subquery(args.get("subquery"), obj)
    k = args.get("k")
    if not isinstance(k, int) or not 1 <= k <= max_k:
        raise ToolCallError(f"k must be an integer in [1, {max_k}]")
    if name == "ProxySearch":
        if args.get("level") not in LEVELS:
            raise ToolCallError(f"level must be one of {list(LEVELS)}")
    else:
        if not isinstance(args.get("node_id"), str) or not args["node_id"]:
            raise ToolCallError("node_id is required")
        nt, et = args.get("node_types"), args.get("edge_types")
        if not nt or any(t not in LEVELS for t in nt):
            raise ToolCallError(f"node_types must be a non-empty subset of {list(LEVELS)}")
        if not et or any(t not in EDGE_TYPES for t in et):
            raise ToolCallError(f"edge_types must be a non-empty subset of {list(EDGE_TYPES)}")
    return {"name": name, "arguments": args}
