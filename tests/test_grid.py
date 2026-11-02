from lantern.grid import Grid, synthesize


def test_synthetic_grid_is_a_forest_of_radial_trees():
    g = synthesize(seed=1)
    heads = [a for a in g.assets.values() if a.parent is None]
    assert len(heads) == len(g.feeders)
    for p in g.premises.values():
        path = g.supply_path(p.id)
        assert path[0].type == "transformer"
        assert path[-1].type == "recloser"
        assert all(a.feeder == p.feeder for a in path)


def test_customers_downstream_counts_every_home_once():
    g = synthesize(seed=2)
    for f in g.feeders:
        head = g.assets[f"{f}-R"]
        assert head.customers_downstream == len(g.feeder_premises(f))


def test_roundtrip_json(tmp_path):
    g = synthesize(seed=3)
    g.save(tmp_path / "g.json")
    h = Grid.load(tmp_path / "g.json")
    assert h.summary() == g.summary()
    assert [a.id for a in h.supply_path("H00001")] == [a.id for a in g.supply_path("H00001")]


def test_address_search_prefers_prefix_matches():
    g = synthesize(seed=4)
    hits = g.find_address("maple")
    assert hits and all("maple" in h.address.lower() for h in hits)
