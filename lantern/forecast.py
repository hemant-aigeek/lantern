"""Weather: turn a daily forecast into storm scenarios.

A ``WeatherSource`` yields one ``DayForecast`` per day. ``OpenMeteo`` pulls
the public Open-Meteo API for the grid's centre; ``SyntheticWeather`` makes a
plausible week from a seed so the system runs offline. Either way the
forecast becomes a ``Storm`` through the same mapping, so nothing downstream
knows or cares where the weather came from.

Forecast skill decays with lead time; ``horizon_confidence`` encodes that so
the assistant can say "day 6 is a guess" instead of pretending otherwise.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass
from typing import Protocol

from .storms import Storm


@dataclass(frozen=True)
class DayForecast:
    date: dt.date
    gust_mph: float
    precip_mm: float
    summary: str

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "gust_mph": round(self.gust_mph, 1), "precip_mm": round(self.precip_mm, 1),
                "summary": self.summary}


class WeatherSource(Protocol):
    name: str

    def days(self) -> list[DayForecast]: ...


def _summary(gust: float, precip: float) -> str:
    wind = "calm" if gust < 25 else "breezy" if gust < 40 else "windy" if gust < 55 else "gale" if gust < 70 else "violent"
    wet = "dry" if precip < 1 else "showers" if precip < 10 else "heavy rain"
    return f"{wind}, {wet}"


class SyntheticWeather:
    """A seeded week with one notable storm, for development and demos."""

    name = "synthetic"

    def __init__(self, seed: int = 3, start: dt.date | None = None, storm_day: int = 2, storm_gust: float = 62.0):
        self.seed, self.storm_day, self.storm_gust = seed, storm_day, storm_gust
        self.start = start or dt.date.today()

    def days(self) -> list[DayForecast]:
        rng = random.Random(self.seed)
        out = []
        for i in range(7):
            if i == self.storm_day:
                gust, precip = self.storm_gust, rng.uniform(15, 35)
            elif i == self.storm_day + 1:
                gust, precip = self.storm_gust * 0.6, rng.uniform(3, 12)
            else:
                gust, precip = rng.uniform(12, 32), rng.choice([0, 0, 0.5, 3, 8])
            out.append(DayForecast(self.start + dt.timedelta(days=i), gust, precip, _summary(gust, precip)))
        return out


class OpenMeteo:
    """Daily max gust and precipitation from api.open-meteo.com (no key needed)."""

    name = "open-meteo"

    def __init__(self, lat: float, lon: float, timeout: float = 8.0):
        self.lat, self.lon, self.timeout = lat, lon, timeout

    def days(self) -> list[DayForecast]:
        import httpx

        r = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": self.lat, "longitude": self.lon, "daily": "wind_gusts_10m_max,precipitation_sum",
                    "wind_speed_unit": "mph", "timezone": "auto", "forecast_days": 7},
            timeout=self.timeout,
        )
        r.raise_for_status()
        d = r.json()["daily"]
        return [DayForecast(dt.date.fromisoformat(t), g or 0.0, p or 0.0, _summary(g or 0.0, p or 0.0))
                for t, g, p in zip(d["time"], d["wind_gusts_10m_max"], d["precipitation_sum"], strict=True)]


def storm_from_day(day: DayForecast, index: int, *, centre=(0.0, 0.0)) -> Storm:
    """A forecast day as a storm scenario.

    Forecast gusts are area averages; a storm's local peaks run above them,
    and saturated ground raises tree-fall. ``wet`` saturates around 25 mm.
    """
    peak = day.gust_mph * 1.15
    wet = 1.0 - math.exp(-day.precip_mm / 25.0)
    return Storm(id=f"day{index}", peak_gust_mph=peak, centre_x=centre[0], centre_y=centre[1], heading_deg=70.0,
                 width_km=12.0, wet=wet, label=f"{day.date.isoformat()} forecast ({day.summary})")


def horizon_confidence(index: int) -> float:
    """How much to trust a forecast ``index`` days out (1.0 today, ~0.45 at day 6)."""
    return max(0.3, 1.0 - 0.09 * index)


def simulated_storm(gust_mph: float, wet: float = 0.4, heading_deg: float = 60.0, width_km: float = 8.0,
                    centre=(0.0, 0.0)) -> Storm:
    return Storm(id=f"sim{int(gust_mph)}", peak_gust_mph=gust_mph, centre_x=centre[0], centre_y=centre[1],
                 heading_deg=heading_deg, width_km=width_km, wet=wet, label=f"simulated {int(gust_mph)} mph storm")
