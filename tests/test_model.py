import math

from lantern import fragility
from lantern.grid import synthesize
from lantern.storms import Storm, simulate_history, true_failure_probability


def test_gust_field_peaks_on_track_and_never_below_floor():
    s = Storm("s", 80, centre_x=0, centre_y=0, heading_deg=0, width_km=5)
    assert math.isclose(s.gust_at(0, 0), 80)
    assert s.gust_at(0, 50) >= 32  # 40% floor
    assert s.gust_at(2, 0) > s.gust_at(0, 2)  # along the track holds up longer than across it


def test_fitted_curves_generalise_to_unseen_storms():
    g = synthesize(seed=7)
    h = simulate_history(g, n_storms=60, seed=11)
    m = fragility.fit(g, h)
    assert m.metrics.cv_roc_auc > 0.85
    assert m.metrics.cv_pr_auc_lift > 5
    assert m.coef["log_gust"] > 0  # more wind, more failures
    assert m.coef["underground"] < 0  # buried lines are safer


def test_fitted_curve_tracks_hidden_truth():
    g = synthesize(seed=7)
    m = fragility.fit(g, simulate_history(g, n_storms=60, seed=11))
    a = next(x for x in g.assets.values() if x.type == "lateral" and not x.underground)
    for gust in (50, 70, 90):
        p, t = m.failure_probability(a, gust, 0.5), true_failure_probability(a, gust, 0.5)
        assert 0.3 * t <= p <= 3 * t, (gust, p, t)


def test_reliability_bins_are_near_diagonal():
    g = synthesize(seed=7)
    m = fragility.fit(g, simulate_history(g, n_storms=60, seed=11))
    top = m.metrics.reliability[-1]
    assert 0.5 * top["observed"] <= top["predicted"] <= 2 * top["observed"]


def test_model_roundtrip(tmp_path):
    g = synthesize(seed=7)
    m = fragility.fit(g, simulate_history(g, n_storms=30, seed=1))
    m.save(tmp_path / "m.pkl")
    m2 = fragility.FragilityModel.load(tmp_path / "m.pkl")
    a = g.assets["F1-P0"]
    assert m2.failure_probability(a, 60, 0.3) == m.failure_probability(a, 60, 0.3)
