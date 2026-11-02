"""HTTP API and the single-page UI.

Scenario selection is uniform across endpoints: ``?day=N`` picks a forecast
day (0 = today), ``?gust=MPH&wet=0..1`` picks a what-if storm. ``crews``
sets the crews available to the restoration simulation.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__, assistant
from .engine import Engine, Scenario
from .forecast import OpenMeteo


def make_engine() -> Engine:
    folder = os.environ.get("LANTERN_DATA")
    weather = None
    if os.environ.get("LANTERN_WEATHER", "synthetic") == "open-meteo":
        lat = float(os.environ.get("LANTERN_LAT", "39.96"))
        lon = float(os.environ.get("LANTERN_LON", "-82.99"))
        weather = OpenMeteo(lat, lon)
    if folder and (Path(folder) / "grid.json").exists():
        return Engine.load(Path(folder), weather)
    return Engine.demo(seed=int(os.environ.get("LANTERN_SEED", "7")), weather=weather)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    premise_id: str | None = None
    audience: str = "customer"
    day: int | None = None
    gust: float | None = None
    wet: float | None = None
    crews: int = 6


def _scenario(day: int | None, gust: float | None, wet: float | None) -> Scenario:
    q: dict = {}
    if gust is not None:
        q["gust"] = gust
        q["wet"] = wet if wet is not None else 0.4
    elif day is not None:
        q["day"] = day
    return Scenario.parse(q)


def build_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="Lantern", version=__version__,
                  description="Storm-outage readiness: who loses power, for how long, and what to do about it.")
    app.state.engine = engine or make_engine()

    def eng(req: Request) -> Engine:
        return req.app.state.engine

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("lantern").joinpath("web/index.html").read_text()

    @app.get("/health")
    def health(req: Request) -> dict:
        e = eng(req)
        return {"ok": True, "version": __version__, "grid": e.grid.name, "weather": e.weather.name}

    @app.get("/forecast")
    def forecast(req: Request) -> dict:
        return {"source": eng(req).weather.name, "days": [d.to_dict() for d in eng(req).forecast()]}

    @app.get("/lookup")
    def lookup(req: Request, q: str = Query(min_length=1)) -> dict:
        return eng(req).lookup(q)

    @app.get("/premises/{premise_id}/risk")
    def premise_risk(req: Request, premise_id: str, day: int | None = None, gust: float | None = None,
                     wet: float | None = None) -> dict:
        _check(eng(req), premise_id)
        return eng(req).outage_risk(premise_id, _scenario(day, gust, wet))

    @app.get("/premises/{premise_id}/restoration")
    def premise_restoration(req: Request, premise_id: str, day: int | None = None, gust: float | None = None,
                            wet: float | None = None, crews: int = Query(6, ge=1, le=200)) -> dict:
        _check(eng(req), premise_id)
        return eng(req).restoration_time(premise_id, _scenario(day, gust, wet), crews)

    @app.get("/premises/{premise_id}/plan")
    def premise_plan(req: Request, premise_id: str, day: int | None = None, gust: float | None = None,
                     wet: float | None = None) -> dict:
        _check(eng(req), premise_id)
        return eng(req).plan(premise_id, _scenario(day, gust, wet))

    @app.get("/premises/{premise_id}/alerts")
    def premise_alerts(req: Request, premise_id: str) -> dict:
        _check(eng(req), premise_id)
        return eng(req).alerts(premise_id)

    @app.post("/chat")
    def chat(req: Request, body: ChatRequest) -> dict:
        if body.premise_id is not None:
            _check(eng(req), body.premise_id)
        if body.audience not in ("customer", "operator"):
            raise HTTPException(400, "audience must be customer or operator")
        r = assistant.answer(eng(req), body.message, premise_id=body.premise_id,
                             scenario=_scenario(body.day, body.gust, body.wet), audience=body.audience, crews=body.crews)
        return r.to_dict()

    @app.get("/ops/briefing")
    def briefing(req: Request, day: int | None = None, gust: float | None = None, wet: float | None = None,
                 crews: int = Query(6, ge=1, le=200)) -> dict:
        return eng(req).briefing(_scenario(day, gust, wet), crews)

    @app.get("/map")
    def map_layer(req: Request, day: int | None = None, gust: float | None = None, wet: float | None = None) -> dict:
        return eng(req).map_layer(_scenario(day, gust, wet))

    @app.get("/model")
    def model_card(req: Request) -> dict:
        return eng(req).model_card()

    @app.get("/audit")
    def audit(req: Request, limit: int = Query(50, ge=1, le=500)) -> dict:
        return {"entries": eng(req).audit.recent(limit), "total": len(eng(req).audit)}

    @app.exception_handler(KeyError)
    async def _key_error(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"error": f"unknown id {exc}"}, status_code=404)

    return app


def _check(e: Engine, premise_id: str) -> None:
    if premise_id not in e.grid.premises:
        raise HTTPException(404, f"unknown premise {premise_id}")
