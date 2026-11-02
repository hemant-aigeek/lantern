"""Restoration time: a crew-dispatch simulation, not a regression.

Utilities rarely have clean time-to-restore labels for individual homes, but
they know how long each kind of repair takes and how many crews they can
field. Lantern therefore *simulates* the restoration: it samples which
equipment fails, turns every failure into a repair job with a crew-hour
estimate, dispatches crews to jobs largest-customer-count first, and reads
off when each home's last broken upstream component is fixed. Repeating this
a few hundred times gives a P10/P50/P90 range and — because the simulation
is explicit — the reason behind it (which job, how many crews, how far down
the queue).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .grid import Grid, Premise
from .storms import Storm

# Mean crew-hours per repair by asset family, and spread (lognormal sigma).
# These are the knobs a utility replaces with its own work-management history.
CREW_HOURS = {"pole": 6.0, "primary": 3.5, "lateral": 2.5, "transformer": 4.0, "fuse": 1.0, "recloser": 3.0,
              "service": 1.5}
HOURS_SIGMA = 0.35
TRAVEL_HOURS = 0.5
PRIORITY_BOOST = 500  # a hospital counts as this many extra customers when ranking jobs


@dataclass
class Job:
    asset_id: str
    type: str
    priority: int
    hours: float
    start: float = 0.0
    end: float = 0.0
    crew: int = -1


@dataclass
class RestorationEstimate:
    premise_id: str | None
    p_outage: float
    crews: int
    runs: int
    hours_p10: float
    hours_p50: float
    hours_p90: float
    bottleneck: dict  # {"type": ..., "share": ...} most common last-fixed job type
    queue_position_p50: float  # how far down the dispatch list the home's jobs sat
    note: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("hours_p10", "hours_p50", "hours_p90", "queue_position_p50"):
            d[k] = round(d[k], 1)
        d["p_outage"] = round(self.p_outage, 4)
        d["source"] = "simulation"
        return d


@dataclass
class AreaEstimate:
    feeder: str | None
    crews: int
    runs: int
    customers: int
    customers_out_p50: float
    crew_hours_p50: float
    hours_to_90pct_p50: float
    hours_to_90pct_p90: float
    crews_for_24h: int
    jobs_by_type_p50: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("customers_out_p50", "crew_hours_p50", "hours_to_90pct_p50", "hours_to_90pct_p90"):
            d[k] = round(d[k], 1)
        d["source"] = "simulation"
        return d


def _sample_jobs(grid: Grid, probs: dict[str, float], rng: random.Random, priority_tx: set[str]) -> list[Job]:
    jobs = []
    for a in grid.assets.values():
        if rng.random() < probs[a.id]:
            hours = CREW_HOURS[a.type] * math.exp(rng.gauss(0, HOURS_SIGMA)) + TRAVEL_HOURS
            prio = a.customers_downstream + (PRIORITY_BOOST if a.id in priority_tx else 0)
            jobs.append(Job(a.id, a.type, prio, hours))
    return jobs


def _dispatch(jobs: list[Job], crews: int, arrival_delay: float) -> None:
    jobs.sort(key=lambda j: -j.priority)
    free = [arrival_delay] * crews
    for j in jobs:
        c = min(range(crews), key=lambda i: free[i])
        j.crew, j.start = c, free[c]
        j.end = j.start + j.hours
        free[c] = j.end


def _priority_transformers(grid: Grid) -> set[str]:
    """Every asset on the path of a priority facility gets the boost."""
    ids: set[str] = set()
    for p in grid.premises.values():
        if p.priority:
            ids.update(a.id for a in grid.supply_path(p.id))
    return ids


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def estimate_premise(grid: Grid, premise: Premise, probs: dict[str, float], *, crews: int = 6,
                     arrival_delay: float = 2.0, runs: int = 300, seed: int = 0) -> RestorationEstimate:
    rng = random.Random(seed)
    path_ids = [a.id for a in grid.supply_path(premise.id)]
    prio = _priority_transformers(grid)
    hours: list[float] = []
    positions: list[float] = []
    last_type: dict[str, int] = {}
    out_runs = 0
    for _ in range(runs):
        jobs = _sample_jobs(grid, probs, rng, prio)
        mine = [j for j in jobs if j.asset_id in path_ids]
        if not mine:
            continue
        out_runs += 1
        _dispatch(jobs, crews, arrival_delay)
        last = max(mine, key=lambda j: j.end)
        hours.append(last.end)
        last_type[last.type] = last_type.get(last.type, 0) + 1
        positions.append(jobs.index(last) + 1)
    p_out = out_runs / runs
    if not hours:
        return RestorationEstimate(premise.id, 0.0, crews, runs, 0, 0, 0, {"type": None, "share": 0}, 0,
                                   note="no outage in any simulated run")
    bt = max(last_type, key=last_type.get)
    return RestorationEstimate(
        premise_id=premise.id, p_outage=p_out, crews=crews, runs=runs,
        hours_p10=_quantile(hours, 0.1), hours_p50=_quantile(hours, 0.5), hours_p90=_quantile(hours, 0.9),
        bottleneck={"type": bt, "share": round(last_type[bt] / len(hours), 2)},
        queue_position_p50=_quantile(positions, 0.5),
        note="hours after the storm passes, given that your home is interrupted",
    )


def estimate_area(grid: Grid, probs: dict[str, float], *, feeder: str | None = None, crews: int = 6,
                  arrival_delay: float = 2.0, runs: int = 120, seed: int = 0) -> AreaEstimate:
    rng = random.Random(seed)
    prio = _priority_transformers(grid)
    prems = grid.feeder_premises(feeder) if feeder else list(grid.premises.values())
    paths = {p.id: [a.id for a in grid.supply_path(p.id)] for p in prems}
    asset_filter = {a.id for a in (grid.feeder_assets(feeder) if feeder else grid.assets.values())}
    outs, crew_hours, t90s, by_type_runs = [], [], [], []
    for _ in range(runs):
        jobs = [j for j in _sample_jobs(grid, probs, rng, prio) if j.asset_id in asset_filter]
        _dispatch(jobs, crews, arrival_delay)
        end_by = {j.asset_id: j.end for j in jobs}
        restore = []
        for path in paths.values():
            ends = [end_by[a] for a in path if a in end_by]
            if ends:
                restore.append(max(ends))
        outs.append(len(restore))
        crew_hours.append(sum(j.hours for j in jobs))
        t90s.append(_quantile(restore, 0.9) if restore else 0.0)
        bt: dict[str, int] = {}
        for j in jobs:
            bt[j.type] = bt.get(j.type, 0) + 1
        by_type_runs.append(bt)
    types = {t for r in by_type_runs for t in r}
    by_type = {t: _quantile([r.get(t, 0) for r in by_type_runs], 0.5) for t in sorted(types)}
    ch50 = _quantile(crew_hours, 0.5)
    # crews needed so the median crew-hours fit in 24 h after arrival
    crews_24 = max(1, math.ceil(ch50 / max(1.0, 24.0 - arrival_delay)))
    return AreaEstimate(
        feeder=feeder, crews=crews, runs=runs, customers=len(prems),
        customers_out_p50=_quantile(outs, 0.5), crew_hours_p50=ch50,
        hours_to_90pct_p50=_quantile(t90s, 0.5), hours_to_90pct_p90=_quantile(t90s, 0.9),
        crews_for_24h=crews_24, jobs_by_type_p50=by_type,
    )


def scenario_for(storm: Storm) -> dict:
    return storm.to_dict()
