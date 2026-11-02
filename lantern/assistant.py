"""The assistant: plain-language answers whose every number comes from a tool.

The flow is fixed: classify the question → call the tools that answer it →
compose a reply from the tool output → (optionally) let a language model
rephrase the reply → check that the rephrased text contains no number the
tools did not produce → return.

Each reply has the same three parts, for customers and operators alike:

- a **direct answer**,
- two or three **things that could change it**,
- a **confidence** score, with what is missing when the score is low.

The language model is a stylist, never a source. If it is absent, slow, or
produces an ungrounded number, the templated reply is returned instead and
the response says so.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .engine import Engine, Scenario

INTENTS = ("risk", "duration", "plan", "alerts", "briefing", "unknown")

_PATTERNS = {
    "risk": r"\b(lose|loose|out|outage|go dark|risk|chance|likely|will i|black ?out)\b",
    "duration": r"\b(how long|when|back on|restore|restoration|eta|until)\b",
    "plan": r"\b(what should|prepare|do i|ready|plan|supplies|generator)\b",
    "alerts": r"\b(alert|notify|notification|text me|qualify|eligib|sign ?up|medical|oxygen)\b",
    "briefing": r"\b(brief|crews?|feeders?|circuits?|status|priorit|dispatch|area|how bad|where)\b",
}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def classify(message: str, audience: str) -> str:
    m = message.lower()
    scores = {k: len(re.findall(p, m)) for k, p in _PATTERNS.items()}
    if audience != "operator":
        scores["briefing"] = 0
    order = ("alerts", "duration", "briefing", "risk", "plan")
    best = max(order, key=lambda k: (scores[k], -order.index(k)))
    return best if scores[best] else ("briefing" if audience == "operator" else "unknown")


@dataclass
class Reply:
    intent: str
    answer: str
    could_change: list[str]
    confidence: dict
    facts: dict = field(default_factory=dict)
    voice: str = "template"  # "template" | "llm" | "template (llm output rejected)"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --- templates ------------------------------------------------------------------

def _pct(p: float) -> str:
    return f"{round(p * 100)}%"


def _hours(h: float) -> str:
    if h < 1.5:
        return "about an hour"
    if h < 36:
        return f"about {round(h)} hours"
    return f"about {round(h / 24, 1)} days"


def compose_risk(r: dict) -> tuple[str, list[str]]:
    lvl, p = r["level"], r["p_outage"]
    sc = r["scenario"]["label"] or "this storm"
    if lvl == "low":
        ans = f"Probably not. For {sc}, the chance of an interruption at your address is {_pct(p)} — low."
    elif lvl == "moderate":
        ans = f"It's possible but not likely. For {sc}, the chance of an interruption at your address is {_pct(p)}."
    else:
        ans = f"There's a real chance. For {sc}, the chance of an interruption at your address is {_pct(p)} — {lvl}."
    w = r["weakest_link"]
    changes = []
    if w:
        changes.append(f"The most exposed part of your route is {w['what']} ({_pct(w['p_fail'])} chance of damage on its own).")
    changes.append(f"Local gusts are expected around {round(r['local_gust_mph'])} mph; "
                   "10 mph more or less would move this a lot.")
    if r["data_completeness"] < 1.0:
        changes.append("Some equipment on your route has no install date on record, so its age is estimated.")
    else:
        changes.append("Saturated ground after heavy rain raises tree-fall, which is the main way lines come down.")
    return ans, changes[:3]


def compose_duration(rt: dict, r: dict) -> tuple[str, list[str]]:
    if rt.get("note", "").startswith("no outage"):
        return ("The simulation didn't produce an outage at your address for this scenario, so there is no restoration "
                "time to estimate."), ["If the storm turns out stronger than forecast, check again closer to the time."]
    ans = (f"If you do lose power, expect it back {_hours(rt['hours_p50'])} after the storm passes. "
           f"The likely range is {_hours(rt['hours_p10'])} to {_hours(rt['hours_p90'])}, with {rt['crews']} crews working.")
    bt = rt["bottleneck"]["type"]
    friendly = {"pole": "a pole replacement", "primary": "a main-line repair", "lateral": "a street-line repair",
                "transformer": "a transformer replacement", "fuse": "a fuse repair", "recloser": "a substation breaker repair",
                "service": "the service wire"}
    changes = [
        f"What usually takes longest is {friendly.get(bt, bt)} upstream of you.",
        f"Your repairs sit around position {round(rt['queue_position_p50'])} in the work queue; more crews shorten the wait.",
        "Crews restore the lines serving the most homes first, so a single-home problem can be last.",
    ]
    return ans, changes


def compose_plan(pl: dict) -> tuple[str, list[str]]:
    lvl = pl["level"]
    steps = pl["steps"]
    lead = {"low": "Risk is low, so a light touch is fine:",
            "moderate": "Risk is moderate; a few simple steps are worth it:",
            "high": "Risk is high; please prepare today:",
            "very high": "Risk is very high; prepare now:"}[lvl]
    ans = lead + " " + " ".join(steps["prepare"])
    changes = [steps["during"][0], steps["during"][1], steps["emergency"][0]]
    return ans, changes


def compose_alerts(al: dict) -> tuple[str, list[str]]:
    if al["eligible"]:
        ans = "Yes — you're eligible for priority storm alerts, because " + al["reason"] + "."
    else:
        ans = "There's no priority flag on file, but anyone can opt in to standard storm alerts for their address."
    return ans, ["Priority alerts go out before a forecast storm, with the restoration outlook for your route.",
                 "If someone at this address relies on powered medical equipment, ask us to add that to your account."]


def compose_briefing(b: dict) -> tuple[str, list[str]]:
    a = b["area"]
    top = b["feeders"][0]
    ans = (f"{b['scenario']['label']}: median {round(a['customers_out_p50'])} of {b['customers_total']} customers "
           f"interrupted, {round(a['crew_hours_p50'])} crew-hours of repairs, 90% restored in "
           f"{_hours(a['hours_to_90pct_p50'])} with {a['crews']} crews. Worst feeder: {top['feeder']} "
           f"({round(top['expected_customers_out'])} expected out, trunk-failure {_pct(top['p_any_trunk_failure'])}).")
    changes = b["actions"][:3] or ["No priority actions triggered at this scenario."]
    return ans, changes


# --- optional language model -------------------------------------------------------

def _numbers(text: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUMBER.findall(text)}


def grounded(candidate: str, source_text: str) -> bool:
    """Every number in ``candidate`` must appear in ``source_text``."""
    return _numbers(candidate) <= _numbers(source_text)


def rephrase(answer: str, could_change: list[str], audience: str) -> str | None:
    """Ask an OpenAI-compatible endpoint to restyle the reply. Returns None if unavailable."""
    base = os.environ.get("LANTERN_LLM_URL")
    if not base:
        return None
    try:
        import httpx

        voice = ("warm, brief, plain customer service; no jargon" if audience == "customer"
                 else "a concise operations briefing for utility staff")
        r = httpx.post(
            base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {os.environ.get('LANTERN_LLM_KEY', '')}"},
            json={"model": os.environ.get("LANTERN_LLM_MODEL", "default"), "temperature": 0.2,
                  "messages": [
                      {"role": "system", "content": f"Rewrite the following in the voice of {voice}. Keep every number "
                                                    "exactly as given. Add no facts. Two short paragraphs at most."},
                      {"role": "user", "content": answer + "\n\nAlso mention: " + " ".join(could_change)},
                  ]},
            timeout=float(os.environ.get("LANTERN_LLM_TIMEOUT", "8")),
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# --- entry point --------------------------------------------------------------------

def answer(engine: Engine, message: str, *, premise_id: str | None, scenario: Scenario, audience: str = "customer",
           crews: int = 6) -> Reply:
    intent = classify(message, audience)
    facts: dict = {}
    if intent == "briefing":
        facts["briefing"] = engine.briefing(scenario, crews=crews)
        ans, changes = compose_briefing(facts["briefing"])
        conf = facts["briefing"]["confidence"]
    elif premise_id is None:
        return Reply("unknown", "Tell me your address first and I can check your home's outlook.",
                     ["Search for your address above, or pick a point on the map."], {"score": 0, "label": "low", "missing": []})
    elif intent == "risk":
        facts["risk"] = engine.outage_risk(premise_id, scenario, audience)
        ans, changes = compose_risk(facts["risk"])
        conf = facts["risk"]["confidence"]
    elif intent == "duration":
        facts["risk"] = engine.outage_risk(premise_id, scenario, audience)
        facts["restoration"] = engine.restoration_time(premise_id, scenario, crews, audience)
        ans, changes = compose_duration(facts["restoration"], facts["risk"])
        conf = facts["restoration"]["confidence"]
    elif intent == "plan":
        facts["plan"] = engine.plan(premise_id, scenario)
        ans, changes = compose_plan(facts["plan"])
        conf = facts["plan"]["confidence"]
    elif intent == "alerts":
        facts["alerts"] = engine.alerts(premise_id)
        ans, changes = compose_alerts(facts["alerts"])
        conf = {"score": 1.0, "label": "high", "missing": []}
    else:
        return Reply("unknown", "I can tell you whether you're likely to lose power, how long it might take to come back, "
                     "what to do to prepare, and whether you qualify for priority alerts.", [], {"score": 1.0, "label": "high",
                                                                                                "missing": []})
    voice = "template"
    styled = rephrase(ans, changes, audience)
    if styled is not None:
        if grounded(styled, ans + " " + " ".join(changes)):
            ans, voice = styled, "llm"
        else:
            voice = "template (llm output rejected: ungrounded number)"
    return Reply(intent, ans, changes, conf, facts, voice)
