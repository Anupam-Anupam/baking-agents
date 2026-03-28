from __future__ import annotations

from dataclasses import dataclass

import requests

from webarena_bake.schemas.types import WebArenaRunRecord


@dataclass
class ObserverDecision:
    score: float
    failure_type: str
    rule_text: str
    confidence: float


def heuristic_observe(record: WebArenaRunRecord) -> ObserverDecision:
    if record.success:
        return ObserverDecision(
            score=max(record.score, 0.8),
            failure_type="none",
            rule_text="Preserve successful action patterns and terminate as soon as objective evidence is collected.",
            confidence=0.9,
        )
    if record.error:
        return ObserverDecision(
            score=0.2,
            failure_type="runtime_error",
            rule_text="Handle page or tool errors by re-synchronizing state before retrying the next minimal action.",
            confidence=0.7,
        )
    if record.steps > 20:
        return ObserverDecision(
            score=0.35,
            failure_type="long_horizon_drift",
            rule_text="If many actions fail to improve progress, restate intent and navigate back to the most relevant page.",
            confidence=0.75,
        )
    return ObserverDecision(
        score=0.45,
        failure_type="action_mismatch",
        rule_text="Map each browser action to an explicit subgoal and avoid exploratory clicks without evidence need.",
        confidence=0.75,
    )


def call_observer_endpoint(record: WebArenaRunRecord, endpoint: str, model: str, api_key: str | None = None) -> ObserverDecision:
    """
    Optional OpenAI-compatible observer call.
    Falls back to heuristic output on request failures.
    """
    prompt = (
        "You are evaluating a web agent trajectory.\n"
        f"Intent: {record.task_intent}\n"
        f"Success: {record.success}\n"
        f"Error: {record.error}\n"
        f"Score: {record.score}\n"
        f"Steps: {record.steps}\n\n"
        "Respond as JSON with keys: score, failure_type, rule_text, confidence."
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 250,
    }
    try:
        response = requests.post(f"{endpoint.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        import json
        import re

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("observer_response_not_json")
        data = json.loads(match.group(0))
        return ObserverDecision(
            score=float(data.get("score", record.score)),
            failure_type=str(data.get("failure_type", "unknown")),
            rule_text=str(data.get("rule_text", "")),
            confidence=float(data.get("confidence", 0.7)),
        )
    except Exception:
        return heuristic_observe(record)

