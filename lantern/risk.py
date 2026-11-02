"""Outage risk along the supply path, and roll-ups by feeder.

A home is interrupted if any piece of equipment between it and the
substation fails, so its outage probability is computed *exactly* from the
per-asset probabilities on its path (assuming independent failures — the
assumption is stated in every response), rather than by averaging a
neighbourhood. Two houses on the same street can get different answers if
one hangs off an underground lateral and the other off an old overhead one,
which is the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fragility import FragilityModel, scenario_probabilities
from .grid import Grid, Premise
from .storms import Storm, category

FRIENDLY = {
    "recloser": "the circuit breaker at the substation",
    "primary": "the main line along the road",
    "fuse": "the fuse where your street branches off",
    "lateral": "the line down your street",
    "transformer": "the transformer serving your home",
    "service": "the wire to your house",
    "pole": "a pole on your route",
}


@dataclass
class PremiseRisk:
    premise_id: str
    storm: Storm
    p_outage: float
    path: list[dict] = field(default_factory=list)  # per-asset contributions, nearest first
    weakest: dict | None = None
    data_completeness: float = 1.0  # share of path assets with known age
    local_gust_mph: float = 0.0

    @property
    def level(self) -> str:
        return level(self.p_outage)

    def to_dict(self) -> dict:
        return {
            "premise_id": self.premise_id,
            "scenario": self.storm.to_dict(),
            "p_outage": round(self.p_outage, 4),
            "level": self.level,
            "local_gust_mph": round(self.local_gust_mph, 1),
            "storm_category": category(self.local_gust_mph),
            "weakest_link": self.weakest,
            "path": self.path,
            "data_completeness": round(self.data_completeness, 3),
            "assumption": "equipment failures are treated as independent along the supply path",
            "source": "model_estimate",
        }


def level(p: float) -> str:
    if p < 0.05:
        return "low"
    if p < 0.2:
        return "moderate"
    if p < 0.5:
        return "high"
    return "very high"


def premise_risk(model: FragilityModel, grid: Grid, premise: Premise, storm: Storm) -> PremiseRisk:
    path = grid.supply_path(premise.id)
    survive = 1.0
    rows = []
    known = 0
    for a in path:
        g = storm.gust_at(a.x, a.y)
        p = model.failure_probability(a, g, storm.wet)
        survive *= 1.0 - p
        known += a.age_known
        rows.append({"type": a.type, "what": FRIENDLY[a.type], "p_fail": round(p, 4), "gust_mph": round(g, 1),
                     "underground": a.underground, "age_known": a.age_known,
                     "drivers": model.drivers(a, g, storm.wet)})
    weakest = max(rows, key=lambda r: r["p_fail"]) if rows else None
    return PremiseRisk(premise_id=premise.id, storm=storm, p_outage=1.0 - survive, path=rows, weakest=weakest,
                       data_completeness=known / max(1, len(path)),
                       local_gust_mph=storm.gust_at(premise.x, premise.y))


@dataclass
class FeederOutlook:
    feeder: str
    customers: int
    expected_customers_out: float
    expected_failed_assets: float
    p_any_trunk_failure: float
    high_risk_assets: int
    priority_facilities: list[dict]

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["expected_customers_out"] = round(self.expected_customers_out, 1)
        d["expected_failed_assets"] = round(self.expected_failed_assets, 2)
        d["p_any_trunk_failure"] = round(self.p_any_trunk_failure, 3)
        d["share_out"] = round(self.expected_customers_out / max(1, self.customers), 3)
        return d


def feeder_outlook(model: FragilityModel, grid: Grid, storm: Storm, high_risk: float = 0.2) -> list[FeederOutlook]:
    probs = scenario_probabilities(model, grid, storm)
    # outage probability per premise, exact along its path
    p_out: dict[str, float] = {}
    for prem in grid.premises.values():
        s = 1.0
        for a in grid.supply_path(prem.id):
            s *= 1.0 - probs[a.id]
        p_out[prem.id] = 1.0 - s
    out = []
    for f in grid.feeders:
        prems = grid.feeder_premises(f)
        assets = list(grid.feeder_assets(f))
        trunk = [a for a in assets if a.type in ("recloser", "primary") or (a.type == "pole" and a.id.count("-") == 1)]
        trunk_survive = 1.0
        for a in trunk:
            trunk_survive *= 1.0 - probs[a.id]
        out.append(FeederOutlook(
            feeder=f, customers=len(prems),
            expected_customers_out=sum(p_out[p.id] for p in prems),
            expected_failed_assets=sum(probs[a.id] for a in assets),
            p_any_trunk_failure=1.0 - trunk_survive,
            high_risk_assets=sum(probs[a.id] >= high_risk for a in assets),
            priority_facilities=[{"id": p.id, "kind": p.kind, "address": p.address, "p_outage": round(p_out[p.id], 3),
                                  "level": level(p_out[p.id])} for p in prems if p.priority],
        ))
    out.sort(key=lambda o: -o.expected_customers_out)
    return out


def map_layer(model: FragilityModel, grid: Grid, storm: Storm) -> dict:
    """Everything a map needs: assets with position, type, and failure probability."""
    probs = scenario_probabilities(model, grid, storm)
    nodes = []
    for a in grid.assets.values():
        parent = grid.assets.get(a.parent) if a.parent else None
        nodes.append({"id": a.id, "type": a.type, "feeder": a.feeder, "x": a.x, "y": a.y,
                      "px": parent.x if parent else None, "py": parent.y if parent else None,
                      "underground": a.underground, "p_fail": round(probs[a.id], 4),
                      "customers": a.customers_downstream})
    homes = [{"id": p.id, "x": p.x, "y": p.y, "kind": p.kind, "feeder": p.feeder} for p in grid.premises.values()]
    return {"scenario": storm.to_dict(), "assets": nodes, "premises": homes}
