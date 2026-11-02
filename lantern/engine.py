"""The engine: one object that owns the grid, the fitted model, the weather,
the sensitive registry and the audit log, and exposes *tools* — the only
functions whose output may be quoted to a customer or an operator.

Every tool returns a plain dict with a ``source`` field, is audit-logged,
and is redacted for its audience before it leaves. The assistant and the
HTTP API are thin layers over these tools.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import fragility, restoration, risk
from .forecast import DayForecast, SyntheticWeather, WeatherSource, horizon_confidence, simulated_storm, storm_from_day
from .grid import Grid, synthesize
from .privacy import AuditLog, SensitiveRegistry, redact
from .storms import History, Storm, simulate_history

FORECAST_TTL_S = 15 * 60


@dataclass(frozen=True)
class Scenario:
    kind: str  # "forecast" | "simulated"
    day: int = 0
    gust_mph: float = 60.0
    wet: float = 0.4

    @classmethod
    def parse(cls, q: dict) -> Scenario:
        if q.get("gust") is not None or q.get("kind") == "simulated":
            return cls("simulated", gust_mph=float(q.get("gust", 60)), wet=float(q.get("wet", 0.4)))
        return cls("forecast", day=max(0, min(6, int(q.get("day", 0)))))

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Engine:
    def __init__(self, grid: Grid, model: fragility.FragilityModel, weather: WeatherSource,
                 registry: SensitiveRegistry | None = None, history: History | None = None):
        self.grid = grid
        self.model = model
        self.weather = weather
        self.registry = registry or SensitiveRegistry()
        self.history = history
        self.audit = AuditLog()
        self._forecast: list[DayForecast] | None = None
        self._forecast_at = 0.0
        self._probs_cache: dict[str, dict[str, float]] = {}

    # -- construction --------------------------------------------------------
    @classmethod
    def demo(cls, seed: int = 7, weather: WeatherSource | None = None) -> Engine:
        grid = synthesize(seed=seed)
        history = simulate_history(grid, seed=seed + 4)
        model = fragility.fit(grid, history)
        reg = SensitiveRegistry()
        # a handful of homes carry a life-support flag; the registry is the only place that knows
        for i, pid in enumerate(sorted(grid.premises)):
            if i % 37 == 0:
                reg.set(pid, life_support=True)
        return cls(grid, model, weather or SyntheticWeather(seed=seed), reg, history)

    @classmethod
    def load(cls, folder: Path, weather: WeatherSource | None = None) -> Engine:
        grid = Grid.load(folder / "grid.json")
        model = fragility.FragilityModel.load(folder / "fragility.pkl")
        return cls(grid, model, weather or SyntheticWeather())

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.grid.save(folder / "grid.json")
        self.model.save(folder / "fragility.pkl")

    # -- weather & scenarios ---------------------------------------------------
    def forecast(self) -> list[DayForecast]:
        if self._forecast is None or time.time() - self._forecast_at > FORECAST_TTL_S:
            try:
                self._forecast = self.weather.days()
            except Exception:  # network trouble: keep the last forecast, or fall back to synthetic
                if self._forecast is None:
                    self._forecast = SyntheticWeather().days()
            self._forecast_at = time.time()
        return self._forecast

    def storm(self, sc: Scenario) -> Storm:
        if sc.kind == "simulated":
            return simulated_storm(sc.gust_mph, sc.wet)
        days = self.forecast()
        return storm_from_day(days[sc.day], sc.day)

    def probabilities(self, storm: Storm) -> dict[str, float]:
        key = f"{storm.id}:{storm.peak_gust_mph:.2f}:{storm.wet:.3f}"
        if key not in self._probs_cache:
            if len(self._probs_cache) > 64:
                self._probs_cache.clear()
            self._probs_cache[key] = fragility.scenario_probabilities(self.model, self.grid, storm)
        return self._probs_cache[key]

    # -- confidence --------------------------------------------------------------
    def confidence(self, sc: Scenario, data_completeness: float = 1.0) -> dict:
        m = self.model.metrics
        calibration = max(0.3, 1.0 - 5.0 * m.cv_brier) if m else 0.6
        horizon = horizon_confidence(sc.day) if sc.kind == "forecast" else 0.75
        score = 0.5 * horizon + 0.3 * data_completeness + 0.2 * calibration
        missing = []
        if sc.kind == "simulated":
            missing.append("this is a what-if storm you chose, not a forecast")
        elif sc.day >= 3:
            missing.append(f"the forecast is {sc.day} days out; wind guidance that far ahead often shifts")
        if data_completeness < 0.8:
            missing.append("install dates are unknown for some equipment on your route")
        if m and m.cv_brier > 0.08:
            missing.append("the model's agreement with past storms is weaker than we would like")
        return {"score": round(score, 2), "label": "high" if score >= 0.75 else "medium" if score >= 0.6 else "low",
                "missing": missing}

    # -- tools -----------------------------------------------------------------
    def _emit(self, tool: str, inputs: dict, out: dict, audience: str) -> dict:
        safe, removed = redact(out, customer_facing=(audience == "customer"))
        self.audit.record(tool, inputs, safe, removed, audience)
        return safe

    def lookup(self, query: str) -> dict:
        hits = self.grid.find_address(query)
        out = {"matches": [{"premise_id": p.id, "address": p.address, "kind": p.kind} for p in hits],
               "source": "records"}
        return self._emit("lookup", {"query": query}, out, "customer")

    def outage_risk(self, premise_id: str, sc: Scenario, audience: str = "customer") -> dict:
        p = self.grid.premises[premise_id]
        storm = self.storm(sc)
        r = risk.premise_risk(self.model, self.grid, p, storm)
        out = r.to_dict()
        out["address"] = p.address
        out["feeder"] = p.feeder
        out["confidence"] = self.confidence(sc, r.data_completeness)
        return self._emit("outage_risk", {"premise_id": premise_id, **sc.to_dict()}, out, audience)

    def restoration_time(self, premise_id: str, sc: Scenario, crews: int = 6, audience: str = "customer") -> dict:
        p = self.grid.premises[premise_id]
        storm = self.storm(sc)
        est = restoration.estimate_premise(self.grid, p, self.probabilities(storm), crews=crews)
        out = est.to_dict()
        out["scenario"] = storm.to_dict()
        out["confidence"] = self.confidence(sc, risk.premise_risk(self.model, self.grid, p, storm).data_completeness)
        return self._emit("restoration_time", {"premise_id": premise_id, "crews": crews, **sc.to_dict()}, out, audience)

    def alerts(self, premise_id: str) -> dict:
        out = {"eligible": self.registry.alert_eligible(premise_id),
               "reason": "a medical-equipment flag is on file for this address" if self.registry.alert_eligible(premise_id)
               else "no medical or priority flag is on file; anyone can opt in to storm alerts",
               "source": "records"}
        return self._emit("alerts", {"premise_id": premise_id}, out, "customer")

    def plan(self, premise_id: str, sc: Scenario) -> dict:
        r = self.outage_risk(premise_id, sc)
        rt = self.restoration_time(premise_id, sc)
        lvl = r["level"]
        steps = {"prepare": [], "during": [], "emergency": []}
        steps["prepare"].append("Charge phones and backup batteries before the wind picks up.")
        if lvl in ("high", "very high"):
            steps["prepare"].append("Fill containers with water and set the fridge and freezer to their coldest settings.")
            steps["prepare"].append(f"Plan for roughly {rt['hours_p50']:.0f} hours without power, "
                                    f"possibly {rt['hours_p90']:.0f}.")
        if self.registry.alert_eligible(premise_id):
            steps["prepare"].append("Confirm your medical equipment's battery runtime and have a place to go if it runs short.")
        steps["during"].append("Stay well away from any downed wire and report it; treat every wire as live.")
        steps["during"].append("Run generators outdoors only, far from windows.")
        steps["emergency"].append("If someone depends on powered medical equipment and it is failing, call emergency services.")
        out = {"level": lvl, "p_outage": r["p_outage"], "restore_hours_p50": rt["hours_p50"], "steps": steps,
               "alerts": self.alerts(premise_id), "confidence": r["confidence"], "source": "model_estimate"}
        return self._emit("plan", {"premise_id": premise_id, **sc.to_dict()}, out, "customer")

    def briefing(self, sc: Scenario, crews: int = 6) -> dict:
        storm = self.storm(sc)
        feeders = [f.to_dict() for f in risk.feeder_outlook(self.model, self.grid, storm)]
        area = restoration.estimate_area(self.grid, self.probabilities(storm), crews=crews).to_dict()
        actions = []
        for f in feeders:
            if f["p_any_trunk_failure"] > 0.5:
                actions.append(f"Stage a crew at {f['feeder']}: trunk failure likely ({f['p_any_trunk_failure']:.0%}).")
            for fac in f["priority_facilities"]:
                if fac["level"] in ("high", "very high"):
                    actions.append(f"Contact {fac['kind']} on {f['feeder']} ({fac['address']}): {fac['level']} outage risk.")
        if area["crews_for_24h"] > crews:
            actions.append(f"Request mutual aid: {area['crews_for_24h']} crews needed to clear the median workload in 24 h; "
                           f"{crews} available.")
        out = {"scenario": storm.to_dict(), "feeders": feeders, "area": area, "actions": actions,
               "customers_total": len(self.grid.premises), "confidence": self.confidence(sc), "source": "model_estimate"}
        return self._emit("briefing", {"crews": crews, **sc.to_dict()}, out, "operator")

    def map_layer(self, sc: Scenario) -> dict:
        return risk.map_layer(self.model, self.grid, self.storm(sc))

    def model_card(self) -> dict:
        m = self.model.metrics
        return {"metrics": m.to_dict() if m else None, "coefficients": {k: round(v, 3) for k, v in self.model.coef.items()},
                "history_storms": len(self.history.storms) if self.history else None, "grid": self.grid.summary(),
                "weather_source": self.weather.name}
