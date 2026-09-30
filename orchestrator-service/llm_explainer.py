"""
llm_explainer.py

Turns an alert's evidence (graph pattern, ML anomaly signals, event timeline) into a
structured, reviewer-facing audit narrative using the local Qwen model served by
llama.cpp's ``llama-server`` (OpenAI-compatible ``/v1/chat/completions``).

The model is only asked to *explain* evidence it is given; it never decides whether an
alert exists. If the server is unreachable or returns unusable output, a deterministic
rule-based explanation is returned instead so the dashboard always has content.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import httpx

logger = logging.getLogger(__name__)

MAX_PROMPT_EVENTS = 40


class RationaleStep(TypedDict):
    step: int
    title: str
    detail: str
    evidence_refs: List[str]


class Explanation(TypedDict):
    summary: str
    breach_tags: List[str]
    rationale: List[RationaleStep]
    recommended_action: str
    model: str
    generated_at: str
    confidence: Optional[float]


SYSTEM_PROMPT = (
    "You are a senior Financial Crime Compliance investigator writing an audit note for a "
    "human reviewer. Use ONLY the evidence provided; never invent names, amounts, dates or IDs. "
    "Cite evidence by the exact IDs given (log_id, tx_id, account_id, emp_id). Be concise, "
    "objective and specific. An ML anomaly score is a percentile, not a probability of guilt. "
    "Respond with a single JSON object and nothing else."
)

RESPONSE_SHAPE = """{
  "summary": "2-3 sentences: who did what, to which accounts, and why it matters",
  "breach_tags": ["short policy / regulation labels, e.g. BSA/AML Structuring, Segregation of Duties"],
  "rationale": [
    {"title": "short step title", "detail": "one or two sentences", "evidence_refs": ["IDs cited"]}
  ],
  "recommended_action": "one clear directive for the reviewer",
  "confidence": 0.0
}"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_reasoning(text: str) -> str:
    """Remove any <think>...</think> block and markdown fences around the JSON."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


class LlamaExplainer:
    """Client for a local llama-server instance generating compliance explanations."""

    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------ health

    async def health(self) -> Dict[str, Any]:
        """Return llama-server health; raises on connection failure."""
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------ prompting

    @staticmethod
    def build_user_prompt(evidence: Dict[str, Any]) -> str:
        """Compact, token-efficient evidence brief (the full graph would not fit usefully)."""
        brief = dict(evidence)
        events = brief.pop("timeline", [])
        if len(events) > MAX_PROMPT_EVENTS:
            brief["timeline_note"] = f"showing first {MAX_PROMPT_EVENTS} of {len(events)} events"
            events = events[:MAX_PROMPT_EVENTS]
        brief["timeline"] = [
            {k: v for k, v in event.items() if v not in (None, "", [])} for event in events
        ]
        return (
            "Write the audit note for this alert.\n\n"
            f"EVIDENCE:\n{json.dumps(brief, indent=1, default=str)}\n\n"
            f"Return JSON exactly in this shape:\n{RESPONSE_SHAPE}\n"
            "Use 3-5 rationale steps in chronological order. "
            "confidence is your 0..1 confidence that this warrants escalation."
        )

    # ------------------------------------------------------------------ generation

    async def generate(self, evidence: Dict[str, Any]) -> Explanation:
        """Call the LLM; raises on transport or parse failure (callers decide on fallback)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.build_user_prompt(evidence)},
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            # Qwen3.x: skip the hidden reasoning pass, we want the JSON directly.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        content = body["choices"][0]["message"].get("content") or ""
        parsed = json.loads(_strip_reasoning(content))
        return self._normalise(parsed)

    def _normalise(self, raw: Dict[str, Any]) -> Explanation:
        steps: List[RationaleStep] = []
        for index, step in enumerate(raw.get("rationale") or []):
            if isinstance(step, str):
                step = {"title": step}
            if not isinstance(step, dict):
                continue
            steps.append({
                "step": index + 1,
                "title": str(step.get("title") or f"Step {index + 1}"),
                "detail": str(step.get("detail") or step.get("description") or ""),
                "evidence_refs": [str(ref) for ref in step.get("evidence_refs") or []],
            })
        confidence = raw.get("confidence")
        try:
            confidence = min(1.0, max(0.0, float(confidence))) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        summary = str(raw.get("summary") or raw.get("executive_summary") or "").strip()
        if not summary:
            raise ValueError("LLM response has no summary")
        return {
            "summary": summary,
            "breach_tags": [str(tag) for tag in raw.get("breach_tags") or []],
            "rationale": steps,
            "recommended_action": str(raw.get("recommended_action") or ""),
            "model": self.model,
            "generated_at": _now(),
            "confidence": confidence,
        }

    # ------------------------------------------------------------------ fallback

    @staticmethod
    def rule_based(evidence: Dict[str, Any], note: str) -> Explanation:
        """Deterministic explanation assembled from detector output."""
        alert = evidence.get("alert", {})
        steps: List[RationaleStep] = []
        for index, finding in enumerate(evidence.get("findings", [])):
            steps.append({
                "step": index + 1,
                "title": finding["title"],
                "detail": finding["detail"],
                "evidence_refs": finding.get("evidence_refs", [])[:8],
            })
        ml = evidence.get("ml", {})
        if ml:
            steps.append({
                "step": len(steps) + 1,
                "title": "ML anomaly triage",
                "detail": (
                    f"{ml.get('anomalous_count', 0)} of {ml.get('scored_count', 0)} transactions flagged "
                    f"by the Isolation Forest (max percentile {ml.get('max_percentile', 0):.2f}). "
                    f"Top drivers: {', '.join(ml.get('top_features', [])) or 'n/a'}."
                ),
                "evidence_refs": ml.get("top_tx_ids", [])[:5],
            })
        return {
            "summary": f"{alert.get('title', 'Alert')}. {alert.get('description', '')} ({note})".strip(),
            "breach_tags": evidence.get("breach_tags", []),
            "rationale": steps,
            "recommended_action": evidence.get("recommended_action", "Manual triage required by L1 analyst."),
            "model": "rule-based",
            "generated_at": _now(),
            "confidence": None,
        }


if __name__ == "__main__":
    import asyncio

    from config import settings

    sample = {
        "alert": {"title": "Insider override before structured wires", "risk_level": "CRITICAL"},
        "employee": {"emp_id": "EMP-A-001", "role": "Senior Compliance Analyst", "is_privileged": True},
        "findings": [{"title": "Override", "detail": "EMP-A-001 overrode an alert on dormant ACC-1",
                      "evidence_refs": ["LOG-A-001"]}],
        "timeline": [
            {"timestamp": "2026-01-01T00:00:00Z", "action_type": "OVERRIDE_ALERT", "emp_id": "EMP-A-001",
             "account_id": "ACC-1", "event_id": "LOG-A-001"},
            {"timestamp": "2026-01-01T00:20:00Z", "amount": 9500, "account_id": "ACC-1",
             "counterparty_account_id": "EXT-9", "event_id": "TX-A-001"},
        ],
    }
    explainer = LlamaExplainer(settings.LLM_BASE_URL, settings.LLM_MODEL, settings.LLM_TIMEOUT_SECONDS)
    print(json.dumps(asyncio.run(explainer.generate(sample)), indent=2))
