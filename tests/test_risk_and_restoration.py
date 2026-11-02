import math

from lantern import restoration, risk
from lantern.engine import Scenario
from lantern.storms import Storm


def test_premise_risk_is_exact_product_along_path(engine, premise_id):
    s = Storm("s", 65, wet=0.5)
    r = risk.premise_risk(engine.model, engine.grid, engine.grid.premises[premise_id], s)
    survive = 1.0
    for row in r.path:
        survive *= 1 - row["p_fail"]
    assert math.isclose(r.p_outage, 1 - survive, abs_tol=1e-3)
    assert r.weakest["p_fail"] == max(x["p_fail"] for x in r.path)


def test_risk_rises_with_wind(engine, premise_id):
    p = engine.grid.premises[premise_id]
    ps = [risk.premise_risk(engine.model, engine.grid, p, Storm("s", g)).p_outage for g in (35, 50, 65, 80)]
    assert ps == sorted(ps) and ps[0] < 0.05 < ps[-1]


def test_feeder_outlook_sums_to_customers(engine):
    out = risk.feeder_outlook(engine.model, engine.grid, Storm("s", 70))
    assert sum(o.customers for o in out) == len(engine.grid.premises)
    for o in out:
        assert 0 <= o.expected_customers_out <= o.customers


def test_more_crews_restore_faster(engine, premise_id):
    s = Storm("s", 70, wet=0.5)
    probs = engine.probabilities(s)
    p = engine.grid.premises[premise_id]
    e2 = restoration.estimate_premise(engine.grid, p, probs, crews=2, runs=200)
    e12 = restoration.estimate_premise(engine.grid, p, probs, crews=12, runs=200)
    assert e12.hours_p50 < e2.hours_p50
    assert e2.hours_p10 <= e2.hours_p50 <= e2.hours_p90


def test_area_estimate_scales_with_storm(engine):
    small = restoration.estimate_area(engine.grid, engine.probabilities(Storm("a", 45)), runs=60)
    big = restoration.estimate_area(engine.grid, engine.probabilities(Storm("b", 80)), runs=60)
    assert big.customers_out_p50 > small.customers_out_p50
    assert big.crew_hours_p50 > small.crew_hours_p50
    assert big.crews_for_24h >= small.crews_for_24h


def test_priority_facilities_are_served_first(engine):
    s = Storm("s", 75, wet=0.6)
    probs = engine.probabilities(s)
    prio = [p for p in engine.grid.premises.values() if p.priority]
    plain = [p for p in engine.grid.premises.values() if not p.priority]
    hp = [restoration.estimate_premise(engine.grid, p, probs, runs=120).hours_p50 for p in prio]
    hn = [restoration.estimate_premise(engine.grid, p, probs, runs=120).hours_p50 for p in plain[::20]]
    assert sum(hp) / len(hp) < sum(hn) / len(hn)


def test_scenario_parsing():
    assert Scenario.parse({"day": "9"}).day == 6
    assert Scenario.parse({"gust": 70}).kind == "simulated"
    assert Scenario.parse({}).kind == "forecast"
