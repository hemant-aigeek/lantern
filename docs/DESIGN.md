# Lantern — design

Lantern tells people on a distribution grid, before a storm arrives, how
likely they are to lose power and how long it would take to get it back —
and tells the people running the grid where to put their crews. This
document explains how it is put together and why.

## 1. The problem, as Lantern sees it

Utilities already have most of what they need to answer "will my lights go
out?": a GIS model of every pole and wire, a log of what broke in past
storms, repair-time standards, and a weather forecast. What they lack is a
way to combine those into an answer that is *specific to one address*,
*honest about its uncertainty*, and *safe to hand to a language model* so it
can be delivered in plain English.

Three design commitments follow:

1. **The supply path is the unit of analysis.** A home is dark if anything
   between it and the substation fails. Every estimate is computed along
   that chain, not averaged over a neighbourhood.
2. **Numbers come from models and simulations; words come from templates or
   a language model that is checked against the numbers.** The language
   model is a stylist, never a source.
3. **Uncertainty is a first-class output.** Every answer carries a
   confidence score, and when the score is low the answer says what is
   missing.

## 2. Architecture

```
            forecast ──► Scenario ──► Storm (gust field over the area)
                                          │
   grid (assets, paths) ──┐               ▼
                          ├──► fragility curves ──► P(fail) per asset
   damage history ────────┘               │
                                          ├──► risk: P(outage) per home, feeder roll-ups
                                          │
                                          └──► restoration: crew-dispatch simulation
                                                      │
                     sensitive registry ──► alerts    ▼
                                          engine.tools ──► redact ──► audit ──► assistant / API / UI
```

| Module | Responsibility |
|--------|----------------|
| `grid.py` | Assets, premises, supply paths, feeder queries; a synthetic generator. |
| `storms.py` | A storm as a spatial gust field; simulated damage history from a hidden truth. |
| `fragility.py` | Per-asset-family failure curves fitted on (asset, storm) pairs; storm-grouped validation; calibration. |
| `risk.py` | Exact outage probability along a path; feeder outlooks; the map layer. |
| `restoration.py` | Monte-Carlo crew dispatch; P10/P50/P90 per home; crew-hours and time-to-90% per area. |
| `forecast.py` | Weather sources (Open-Meteo, synthetic) → storm scenarios; forecast-horizon confidence. |
| `privacy.py` | The sensitive registry, boundary redaction, audit log. |
| `engine.py` | Owns everything above; exposes audited, redacted *tools*; computes confidence. |
| `assistant.py` | Intent → tools → templated reply → optional LLM restyle → grounding check. |
| `api.py`, `web/index.html` | FastAPI endpoints and a single-page customer/operations UI. |

## 3. Fragility, not classification

The obvious approach — train a classifier on "was this asset damaged in this
storm" — produces a number that is hard to argue with because it is hard to
read. Lantern instead fits **fragility curves**: for each asset family, the
probability of failure as a function of the gust the asset actually sees,
shifted by age, canopy cover, construction class, conductor length and
ground saturation. The functional form is logistic in log-gust; the fit is a
regularised logistic regression on every (asset, storm) pair in the history.

Two details matter more than the model class:

- **Local gust, not storm category.** A storm is a field (`Storm.gust_at`)
  — peak along its track, fading across it — so an exposed lateral on the
  track and a sheltered one across town are scored on different winds in
  the same storm. Real deployments replace the analytic field with gridded
  forecast or reanalysis winds.
- **Validation is grouped by storm.** The question is how the model does on a
  storm it has not seen, so cross-validation folds hold out whole storms.
  Lantern reports PR-AUC lift over the base rate, ROC-AUC, Brier score and a
  reliability table. Accuracy is not reported: at sub-percent base rates it
  is meaningless.

On the synthetic grid the fitted curve tracks the hidden truth within a
factor of ~2 across 50–90 mph, which is the test (`test_fitted_curve_tracks_hidden_truth`).

### 3.1 Why a hidden truth

The synthetic history is generated from a fragility function the model is
never shown. That makes calibration *measurable* — the fitted curve can be
compared to what actually generated the data — which is impossible with a
real history and is exactly the property worth having while the pipeline is
being built. Swapping in real data changes nothing downstream.

## 4. Risk along the path

For a home with supply path `a₁ … aₙ` and per-asset failure probabilities
`pᵢ`:

```
P(outage) = 1 − Π (1 − pᵢ)
```

The independence assumption is stated in every response. It is conservative
in one direction (correlated failures on a trunk make the tail worse) and
optimistic in another (a single tree can take out two adjacent spans), and
replacing it with a copula is a contained change in `risk.premise_risk`.

The response also names the **weakest link** (the path asset with the
highest `pᵢ`), its top **drivers** in log-odds, and the share of path assets
with a known install date (**data completeness**), which feeds confidence.

Feeder roll-ups sum the per-home probabilities (expected customers out),
sum per-asset probabilities (expected failed assets), and compute the
probability that *anything on the trunk* fails — the number an operator
uses to decide where to pre-stage a crew.

## 5. Restoration as a simulation

Utilities rarely have clean per-home time-to-restore labels, but they do know
repair-time standards and crew counts. So Lantern simulates:

1. Sample which assets fail (Bernoulli on each `pᵢ`).
2. Each failure becomes a job with lognormal crew-hours around its family's
   standard, plus travel.
3. Dispatch: jobs are ranked by customers downstream (with a large bonus for
   anything on a priority facility's path), and each job goes to the
   earliest-free crew after an arrival delay.
4. A home is restored when the *last* failed asset on its path is repaired.

Repeating this a few hundred times gives P10/P50/P90 per home, conditional
on being interrupted, together with *why*: which job type is usually the
last one fixed and where the home's jobs sit in the queue. The area
estimate gives median customers out, crew-hours, time to 90% restored, and
the crews needed to clear the median workload within 24 hours — which is
the mutual-aid question.

The simulation is deliberately simple (no travel geography, no switching,
no partial restoration by sectionalising) and every simplification lives in
one function so it can be replaced.

## 6. Confidence

Each answer carries `{score, label, missing}`:

```
score = 0.5 · horizon + 0.3 · data_completeness + 0.2 · calibration
```

- `horizon` decays with forecast lead time (1.0 today, ~0.45 at day 6) and
  is fixed at 0.75 for a what-if storm.
- `data_completeness` is the share of path assets with known age.
- `calibration` is derived from the model's cross-validated Brier score.

Below 0.6 the reply lists what is missing in plain words: "the forecast is
5 days out", "install dates are unknown for some equipment on your route",
"this is a what-if storm you chose, not a forecast".

## 7. The assistant

The pipeline is fixed and short:

```
classify(message) → call tools → compose template → [LLM restyle] → grounding check → reply
```

- **Intents** are matched by keyword patterns with a fixed tie-break order;
  there is no LLM in the loop for routing. Operators get a `briefing`
  intent; customers do not.
- **Every reply** has a direct answer, two or three things that could change
  it, and the confidence block, in both voices.
- **The language model is optional** (`LANTERN_LLM_URL`, any
  OpenAI-compatible endpoint). It receives the templated text and is asked
  only to restyle it. Its output is accepted only if every number in it
  also appears in the template; otherwise the template is returned and the
  response's `voice` field says so. A test monkeypatches the model to
  invent "0% chance" and asserts the fallback.

## 8. Privacy

Two structural mechanisms, both tested:

- **The sensitive registry.** Life-support, threat and protected-contact
  flags live in `SensitiveRegistry`, which the grid, the model, the
  simulation and the assistant never receive. The only thing that leaves it
  is a boolean *eligible for priority alerts*.
- **Redaction at the boundary.** Every tool result passes through `redact`
  before it leaves the engine: internal asset ids, raw coordinates and any
  sensitive key are removed for customer audiences, and the audit log
  records the tool, a hash of its input, the fields that went out and the
  fields that were removed.

## 9. Bringing real data

Lantern's grid and history are plain dataclasses, so an adapter is a
function that returns a `Grid` and a `History`:

- GIS export → `Asset` rows: type, feeder, position, upstream id, age,
  canopy class, construction class, conductor length, underground flag.
- OMS/work-management history → `DamageRecord` rows: storm id, asset id,
  local gust (from reanalysis at the asset's location and the storm's time).
- Crew-hour standards → `restoration.CREW_HOURS`.
- Forecast → `OpenMeteo` (already there) or a gridded source behind the
  `WeatherSource` protocol.

Nothing in `engine`, `assistant`, `api` or the UI changes.

## 10. Non-goals and limits

- Not an outage-management system: Lantern predicts and explains; it does
  not ingest live outage tickets or dispatch real crews.
- Not a weather model: it consumes a forecast and says how much to trust it.
- Independence along the path and a single analytic gust field are stated
  simplifications (§4, §3).
- The synthetic grid is for building and demonstrating the pipeline; its
  numbers describe no real utility.
