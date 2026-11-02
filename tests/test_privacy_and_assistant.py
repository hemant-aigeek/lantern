import json

from lantern import assistant
from lantern.engine import Scenario
from lantern.privacy import SENSITIVE_KEYS, SensitiveRegistry, redact

FORBIDDEN = SENSITIVE_KEYS | {"x", "y", "px", "py", "transformer", "parent"}


def _keys(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(k)
            _keys(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, acc)
    return acc


def test_customer_outputs_carry_no_internal_ids_or_coordinates(engine, premise_id):
    sc = Scenario("forecast", day=2)
    for out in (engine.outage_risk(premise_id, sc), engine.restoration_time(premise_id, sc), engine.plan(premise_id, sc)):
        assert not (_keys(out, set()) & FORBIDDEN)
        assert "F1-" not in json.dumps(out) and "F2-" not in json.dumps(out)


def test_sensitive_flags_never_reach_the_grid_or_model(engine):
    assert not any(hasattr(p, k) for p in engine.grid.premises.values() for k in SENSITIVE_KEYS)
    flagged = [pid for pid in engine.grid.premises if engine.registry.alert_eligible(pid)]
    assert flagged
    assert "registry" not in json.dumps(engine.model_card())


def test_redact_reports_what_it_removed():
    payload = {"p": 0.1, "life_support": True, "x": 1.0, "nested": [{"asset_id": "F1-P0", "ok": 1}]}
    safe, removed = redact(payload)
    assert safe == {"p": 0.1, "nested": [{"ok": 1}]}
    assert set(removed) == {"life_support", "x", "nested[].asset_id"}


def test_registry_only_exposes_eligibility():
    r = SensitiveRegistry()
    r.set("H1", life_support=True, threat=True)
    assert r.alert_eligible("H1") and not r.alert_eligible("H2")
    assert "threat" not in dir(r)


def test_audit_log_records_every_tool_call(engine, premise_id):
    before = len(engine.audit)
    engine.outage_risk(premise_id, Scenario("forecast", day=1))
    engine.alerts(premise_id)
    entries = engine.audit.recent(2)
    assert len(engine.audit) == before + 2
    assert [e["tool"] for e in entries] == ["outage_risk", "alerts"]
    assert entries[0]["audience"] == "customer"


def test_intent_classification():
    c = assistant.classify
    assert c("Will I lose power?", "customer") == "risk"
    assert c("How long until my power is back?", "customer") == "duration"
    assert c("What should I do?", "customer") == "plan"
    assert c("Do I qualify for alerts?", "customer") == "alerts"
    assert c("how bad is F2", "operator") == "briefing"
    assert c("hello there", "customer") == "unknown"


def test_every_number_in_an_answer_comes_from_the_tools(engine, premise_id):
    for q in ("Will I lose power?", "How long until my power is back?", "What should I do?"):
        r = assistant.answer(engine, q, premise_id=premise_id, scenario=Scenario("forecast", day=2))
        facts = json.dumps(r.facts)
        for n in assistant._numbers(r.answer + " " + " ".join(r.could_change)):
            assert n in facts or n in {"10", "1", "24"}, (q, n)  # fixed phrases: "10 mph", "an hour"
        assert set(r.confidence) == {"score", "label", "missing"}


def test_grounding_check_rejects_invented_numbers():
    src = "The chance is 41%. Expect about 7 hours."
    assert assistant.grounded("Roughly 41% — about 7 hours.", src)
    assert not assistant.grounded("Roughly 45% — about 7 hours.", src)


def test_llm_rejection_falls_back_to_template(engine, premise_id, monkeypatch):
    monkeypatch.setattr(assistant, "rephrase", lambda *a, **k: "Totally safe, 0% chance, back in 2 minutes.")
    r = assistant.answer(engine, "Will I lose power?", premise_id=premise_id, scenario=Scenario("forecast", day=2))
    assert r.voice.startswith("template (llm output rejected")
    assert "0% chance" not in r.answer


def test_operator_briefing_without_premise(engine):
    r = assistant.answer(engine, "status", premise_id=None, scenario=Scenario("simulated", gust_mph=70), audience="operator")
    assert r.intent == "briefing" and "customers interrupted" in r.answer
    assert r.confidence["missing"]  # a what-if storm says so


def test_low_confidence_lists_what_is_missing(engine):
    c = engine.confidence(Scenario("forecast", day=6), data_completeness=0.5)
    assert c["label"] == "low" and len(c["missing"]) >= 2
