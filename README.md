# Lantern

**Storm-outage readiness for distribution utilities.** Before a storm,
Lantern tells every home on the grid how likely it is to lose power and how
long it would take to get it back, and tells operators where the damage
will land and how many crews it will take — in plain language, with every
number traced to a model and never to a language model.

```
$ lantern ask "Will I lose power?" --address Maple
[1200 Maple Ln]
There's a real chance. For 2026-11-03 forecast (gale, heavy rain), the chance of an
interruption at your address is 41% — high.
  - The most exposed part of your route is the main line along the road (9% chance of damage on its own).
  - Local gusts are expected around 65 mph; 10 mph more or less would move this a lot.
  - Saturated ground after heavy rain raises tree-fall, which is the main way lines come down.
  confidence 90% (high)

$ lantern ask "status" --audience operator --gust 70
simulated 70 mph storm: median 395 of 840 customers interrupted, 88 crew-hours of repairs,
90% restored in about 15 hours with 6 crews. Worst feeder: F2 (106 expected out, trunk-failure 35%).
  - Contact water on F2 (628 Beacon Way (water)): high outage risk.
  - Contact fire on F2 (104 Pine Ave (fire)): very high outage risk.
  confidence 87% (high); missing: this is a what-if storm you chose, not a forecast
```

## What it does

- **Per-address outage probability**, computed exactly along the home's
  supply path from per-asset *fragility curves* — failure probability as a
  function of the gust each pole, span, fuse and transformer actually sees,
  shifted by age, canopy, construction class and ground saturation. The
  answer names the weakest link on the route and what drives it.
- **Restoration time as a simulation**, not a regression: sample what
  breaks, turn each failure into a repair job, dispatch crews
  largest-outage-first with priority facilities boosted, and read off when
  each home's last upstream fault is fixed. You get P10/P50/P90 hours and
  the reason (which repair, how deep in the queue).
- **An operations briefing** per feeder: expected customers out, trunk
  failure probability, high-risk assets, priority facilities at risk,
  crew-hours, time to 90% restored, and the crews needed to clear the work
  in 24 hours.
- **A grounded assistant** with one shape for every reply — a direct answer,
  two or three things that could change it, and a confidence score that
  lists what is missing when it is low. An LLM may restyle the text
  (optional, any OpenAI-compatible endpoint); its output is rejected if it
  contains a number the tools did not produce.
- **Privacy by structure**: medical and threat flags live in a registry the
  model and assistant never see; internal ids and coordinates are stripped at
  the boundary; every tool call is audit-logged with what went out and what
  was removed.
- **Runs with no private data.** A synthetic grid and storm history are
  generated from a seed, so the whole pipeline — fitting, validation,
  simulation, UI — works on a laptop. Adapters bring in real GIS, outage
  history and forecasts without touching the rest.

Read the [design document](docs/DESIGN.md) for the reasoning.

## Install and run

```sh
pip install -e ".[dev]"
lantern serve                    # http://127.0.0.1:8080 — customer and operations views
```

The UI has a storm scenario picker (a forecast day, or a what-if gust and
ground saturation), a map of the grid coloured by failure probability, an
address search with chat on the **My home** tab, and feeder KPIs, suggested
actions and a staff-voice chat on the **Operations** tab.

Use a live forecast instead of the synthetic week:

```sh
LANTERN_WEATHER=open-meteo LANTERN_LAT=39.96 LANTERN_LON=-82.99 lantern serve
```

Let a language model phrase the replies (numbers are still checked):

```sh
LANTERN_LLM_URL=https://api.example.com/v1 LANTERN_LLM_KEY=... LANTERN_LLM_MODEL=... lantern serve
```

Fit and save a model to a folder, then serve from it:

```sh
lantern build ./data --seed 11
lantern serve --data ./data
```

## API

Scenario selection is the same everywhere: `?day=N` (forecast day, 0 = today)
or `?gust=MPH&wet=0..1` (what-if storm); `crews=N` for the simulation.

| Route | |
|-------|-|
| `GET /forecast` | Seven days of gust and precipitation, with the source. |
| `GET /lookup?q=` | Address search → premise ids. |
| `GET /premises/{id}/risk` | Outage probability, level, weakest link, per-asset path, confidence. |
| `GET /premises/{id}/restoration` | P10/P50/P90 hours, bottleneck, queue position, confidence. |
| `GET /premises/{id}/plan` | Prepare / during / emergency steps and alert eligibility. |
| `GET /premises/{id}/alerts` | Priority-alert eligibility (the only output of the sensitive registry). |
| `POST /chat` | `{message, premise_id?, audience, day|gust, crews}` → answer, could-change list, confidence, facts, voice. |
| `GET /ops/briefing` | Feeder table, area simulation, suggested actions. |
| `GET /map` | Assets with positions and failure probabilities; premises. |
| `GET /model` | Cross-validated metrics, reliability table, coefficients. |
| `GET /audit` | Recent tool calls: tool, input hash, fields out, fields removed, audience. |

## How the numbers are made

| Question | Method | Output |
|----------|--------|--------|
| Will this asset fail? | Logistic fragility curve in log-gust, fitted on (asset, storm) pairs; validation grouped by storm (PR-AUC lift, ROC-AUC, Brier, reliability) | `p_fail` |
| Will this home lose power? | `1 − Π(1 − p_fail)` along the supply path | `p_outage`, weakest link, drivers |
| How long to restore? | Monte-Carlo crew dispatch over sampled failures, largest-outage-first with priority boost | P10/P50/P90, bottleneck |
| How much should I trust this? | `0.5·horizon + 0.3·data completeness + 0.2·calibration` | score, label, what's missing |

## Tests

```sh
pytest
```

Thirty-one tests cover the grid's structure, the storm field, model
generalisation to held-out storms and agreement with the hidden truth,
calibration, exactness of path risk, monotonicity of risk in wind and of
restoration in crews, priority-facility ordering, redaction, the registry's
isolation, the audit log, intent routing, the grounding check and its
fallback, and every API route.

## Scope

Lantern predicts and explains; it is not an outage-management system and
does not dispatch anyone. Failures along a path are treated as independent
and the storm is an analytic gust field — both stated simplifications with
one place each to replace them. The synthetic grid describes no real
utility.

## License

Apache-2.0.
