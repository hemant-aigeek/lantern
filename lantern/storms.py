"""Storms: a wind-gust field over the service area, and simulated history.

A ``Storm`` is described by a peak gust, a track (centre and direction) and
how quickly it fades with distance from the track. ``gust_at`` gives the
local gust for any asset, so the *same* storm stresses an exposed ridge
lateral more than a sheltered trunk.

``simulate_history`` plays a set of past storms over a grid using a hidden
*true* fragility (the thing the model must learn) and returns the damage
records a utility's outage-management system would have logged. Lantern
never trains on this truth directly — only on the records — so the model's
calibration can be measured honestly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .grid import Asset, Grid


@dataclass(frozen=True)
class Storm:
    id: str
    peak_gust_mph: float
    centre_x: float = 0.0
    centre_y: float = 0.0
    heading_deg: float = 60.0  # direction the track runs
    width_km: float = 6.0  # how far from the track the peak holds up
    wet: float = 0.3  # 0 dry .. 1 saturated ground / ice loading
    label: str = ""

    def gust_at(self, x: float, y: float) -> float:
        """Local gust: peak along the track, fading across it, never below 40%."""
        th = math.radians(self.heading_deg)
        dx, dy = x - self.centre_x, y - self.centre_y
        across = abs(-math.sin(th) * dx + math.cos(th) * dy)
        along = abs(math.cos(th) * dx + math.sin(th) * dy)
        fade = math.exp(-((across / self.width_km) ** 2)) * math.exp(-((along / (3 * self.width_km)) ** 2))
        return self.peak_gust_mph * (0.4 + 0.6 * fade)

    def to_dict(self) -> dict:
        return {"id": self.id, "peak_gust_mph": round(self.peak_gust_mph, 1), "centre": [self.centre_x, self.centre_y],
                "heading_deg": self.heading_deg, "width_km": self.width_km, "wet": self.wet, "label": self.label}


def category(peak_gust_mph: float) -> str:
    if peak_gust_mph < 30:
        return "calm"
    if peak_gust_mph < 45:
        return "breezy"
    if peak_gust_mph < 60:
        return "strong"
    if peak_gust_mph < 75:
        return "severe"
    return "extreme"


@dataclass
class DamageRecord:
    storm_id: str
    asset_id: str
    gust_mph: float


@dataclass
class History:
    storms: list[Storm] = field(default_factory=list)
    records: list[DamageRecord] = field(default_factory=list)

    def damaged_in(self, storm_id: str) -> set[str]:
        return {r.asset_id for r in self.records if r.storm_id == storm_id}


# --- hidden truth, used only to simulate records -----------------------------

_TRUE_BASE = {"pole": -9.4, "primary": -8.9, "lateral": -8.6, "transformer": -10.0, "fuse": -9.3,
              "recloser": -10.8, "service": -8.3}


def true_failure_probability(a: Asset, gust_mph: float, wet: float) -> float:
    """What actually happens to an asset at a given gust. Not available to the model."""
    if a.underground:
        return 0.0005 + 0.002 * wet
    z = _TRUE_BASE[a.type] + 3.6 * math.log(max(gust_mph, 1.0) / 40.0) * 2.0
    z += 0.025 * a.age_years + 1.8 * a.vegetation + 0.8 * wet
    if a.type == "pole":
        z += 0.35 * (a.pole_class - 3)
    if a.type in ("primary", "lateral"):
        z += 0.6 * a.length_km
    return 1.0 / (1.0 + math.exp(-z))


def random_storm(rng: random.Random, sid: str, peak: float | None = None) -> Storm:
    peak = peak if peak is not None else min(110.0, max(25.0, rng.lognormvariate(math.log(45), 0.35)))
    return Storm(id=sid, peak_gust_mph=peak, centre_x=rng.uniform(-3, 3), centre_y=rng.uniform(-3, 3),
                 heading_deg=rng.uniform(0, 180), width_km=rng.uniform(3, 9), wet=rng.random())


def simulate_history(grid: Grid, n_storms: int = 60, seed: int = 11) -> History:
    rng = random.Random(seed)
    hist = History()
    for i in range(n_storms):
        s = random_storm(rng, f"S{i + 1:03d}")
        hist.storms.append(s)
        for a in grid.assets.values():
            g = s.gust_at(a.x, a.y)
            if rng.random() < true_failure_probability(a, g, s.wet):
                hist.records.append(DamageRecord(s.id, a.id, g))
    return hist
