from fastapi.testclient import TestClient

from lantern.api import build_app


def _client(engine):
    return TestClient(build_app(engine))


def test_health_and_forecast(engine):
    c = _client(engine)
    assert c.get("/health").json()["ok"]
    days = c.get("/forecast").json()["days"]
    assert len(days) == 7 and {"date", "gust_mph", "precip_mm", "summary"} <= set(days[0])


def test_lookup_risk_restoration_plan(engine):
    c = _client(engine)
    pid = c.get("/lookup", params={"q": "Maple"}).json()["matches"][0]["premise_id"]
    r = c.get(f"/premises/{pid}/risk", params={"day": 2}).json()
    assert 0 <= r["p_outage"] <= 1 and r["source"] == "model_estimate"
    rt = c.get(f"/premises/{pid}/restoration", params={"gust": 70, "crews": 4}).json()
    assert rt["crews"] == 4 and rt["source"] == "simulation"
    pl = c.get(f"/premises/{pid}/plan", params={"day": 2}).json()
    assert set(pl["steps"]) == {"prepare", "during", "emergency"}
    assert c.get("/premises/NOPE/risk").status_code == 404


def test_chat_both_audiences(engine):
    c = _client(engine)
    pid = c.get("/lookup", params={"q": "Cedar"}).json()["matches"][0]["premise_id"]
    r = c.post("/chat", json={"message": "will I lose power", "premise_id": pid, "day": 2}).json()
    assert r["intent"] == "risk" and r["answer"]
    o = c.post("/chat", json={"message": "briefing", "audience": "operator", "gust": 65, "crews": 8}).json()
    assert o["intent"] == "briefing"
    assert c.post("/chat", json={"message": "x", "audience": "nobody"}).status_code == 400


def test_briefing_map_model_audit(engine):
    c = _client(engine)
    b = c.get("/ops/briefing", params={"gust": 70}).json()
    assert len(b["feeders"]) == len(engine.grid.feeders) and "actions" in b
    m = c.get("/map", params={"day": 2}).json()
    assert len(m["assets"]) == len(engine.grid.assets)
    assert c.get("/model").json()["metrics"]["cv_roc_auc"] > 0.8
    assert c.get("/audit").json()["total"] > 0
    assert "<title>Lantern</title>" in c.get("/").text
