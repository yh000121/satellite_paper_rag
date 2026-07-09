from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def build_rule_library(
    paper_id: str,
    title: str,
    rules: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unique_rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        if rule.get("answer_type") == "insufficient_evidence":
            continue
        rule_copy = copy.deepcopy(rule)
        dedupe_key = _rule_fingerprint(rule_copy)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        if not rule_copy.get("rule_id"):
            rule_copy["rule_id"] = f"{paper_id}_rule_{len(unique_rules) + 1:04d}"
        unique_rules.append(rule_copy)
    return {
        "paper_id": paper_id,
        "title": title,
        "rule_count": len(unique_rules),
        "metadata": dict(metadata or {}),
        "rules": unique_rules,
    }


def write_rule_library(rule_library: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rule_library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_rule_library(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Rule library must be a JSON object: {path}")
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"Rule library must contain a 'rules' list: {path}")
    return payload


def _rule_fingerprint(rule: dict[str, Any]) -> str:
    comparable = copy.deepcopy(rule)
    comparable.pop("rule_id", None)
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
