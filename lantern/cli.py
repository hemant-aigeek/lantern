"""Command line: ``lantern serve`` runs the API and UI; ``lantern build`` fits a
model from a grid and a damage history (or synthesises both); ``lantern ask``
answers one question from the terminal."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from . import __version__


def _serve(args) -> None:
    import uvicorn

    from .api import build_app

    if args.data:
        os.environ["LANTERN_DATA"] = args.data
    if args.weather:
        os.environ["LANTERN_WEATHER"] = args.weather
    app = build_app()
    print(f"lantern {__version__} · {app.state.engine.grid.name} · http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _build(args) -> None:
    from .engine import Engine

    e = Engine.demo(seed=args.seed)
    e.save(Path(args.out))
    m = e.model.metrics
    print(json.dumps({"saved": args.out, "grid": e.grid.summary(),
                      "model": {k: v for k, v in m.to_dict().items() if k != "reliability"}}, indent=2))


def _ask(args) -> None:
    from . import assistant
    from .engine import Engine, Scenario

    e = Engine.load(Path(args.data)) if args.data else Engine.demo()
    pid = None
    if args.address:
        hits = e.lookup(args.address)["matches"]
        if not hits:
            raise SystemExit(f"no address matches {args.address!r}")
        pid = hits[0]["premise_id"]
        print(f"[{hits[0]['address']}]")
    sc = Scenario.parse({"gust": args.gust} if args.gust else {"day": args.day})
    r = assistant.answer(e, args.question, premise_id=pid, scenario=sc, audience=args.audience, crews=args.crews)
    print(r.answer)
    for c in r.could_change:
        print(f"  - {c}")
    print(f"  confidence {r.confidence['score']:.0%} ({r.confidence['label']})"
          + (f"; missing: {'; '.join(r.confidence['missing'])}" if r.confidence["missing"] else ""))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lantern", description="storm-outage readiness for distribution utilities")
    ap.add_argument("--version", action="version", version=f"lantern {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the API and web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--data", help="folder with grid.json + fragility.pkl (default: synthetic demo)")
    s.add_argument("--weather", choices=["synthetic", "open-meteo"])
    s.set_defaults(fn=_serve)

    b = sub.add_parser("build", help="synthesise a grid and history, fit the model, save to a folder")
    b.add_argument("out")
    b.add_argument("--seed", type=int, default=7)
    b.set_defaults(fn=_build)

    a = sub.add_parser("ask", help="ask one question from the terminal")
    a.add_argument("question")
    a.add_argument("--address", help="address fragment to look up")
    a.add_argument("--day", type=int, default=2)
    a.add_argument("--gust", type=float, help="use a what-if storm with this peak gust instead of a forecast day")
    a.add_argument("--audience", choices=["customer", "operator"], default="customer")
    a.add_argument("--crews", type=int, default=6)
    a.add_argument("--data")
    a.set_defaults(fn=_ask)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    args.fn(args)


if __name__ == "__main__":
    main()
