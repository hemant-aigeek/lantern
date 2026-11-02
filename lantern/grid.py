"""The grid model: a radial distribution network and the homes it serves.

Lantern reasons about outages along the **supply path** — the chain of
equipment between a home and its substation. A home is dark if *anything*
on that chain fails, so the path, not the map, is the unit of analysis.

The model is deliberately small and generic so it can be filled from any
utility's GIS/OMS export (see ``adapters`` in the design doc) or generated
synthetically for development and demos (``synthesize``).

Coordinates are kilometres on a local plane; the synthetic generator centres
the substation at the origin.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

ASSET_TYPES = ("recloser", "primary", "fuse", "lateral", "transformer", "service", "pole")
PROTECTIVE = {"recloser", "fuse"}


@dataclass
class Asset:
    id: str
    type: str  # one of ASSET_TYPES
    feeder: str
    x: float
    y: float
    parent: str | None  # upstream asset id; None for the feeder head
    age_years: float
    vegetation: float  # 0 (clear) .. 1 (heavy canopy)
    underground: bool = False
    length_km: float = 0.0  # conductors only
    pole_class: int = 4  # poles: ANSI class, lower is stronger
    kva: float = 0.0  # transformers
    customers_downstream: int = 0
    age_known: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Premise:
    id: str
    address: str
    feeder: str
    transformer: str  # asset id the service hangs from
    x: float
    y: float
    kind: str = "residential"  # residential | commercial | hospital | fire | water | shelter
    # NOTE: medical/life-support flags are *not* here on purpose; see privacy.py

    @property
    def priority(self) -> bool:
        return self.kind in PRIORITY_KINDS

    def to_dict(self) -> dict:
        return asdict(self)


PRIORITY_KINDS = ("hospital", "fire", "water", "shelter")


@dataclass
class Grid:
    name: str
    assets: dict[str, Asset] = field(default_factory=dict)
    premises: dict[str, Premise] = field(default_factory=dict)
    feeders: list[str] = field(default_factory=list)
    centre_lat: float = 0.0
    centre_lon: float = 0.0
    _children: dict[str, list[str]] = field(default_factory=dict, repr=False)

    # -- structure ---------------------------------------------------------
    def add(self, a: Asset) -> None:
        self.assets[a.id] = a
        if a.parent:
            self._children.setdefault(a.parent, []).append(a.id)
        if a.feeder not in self.feeders:
            self.feeders.append(a.feeder)

    def children(self, asset_id: str) -> list[str]:
        return self._children.get(asset_id, [])

    def supply_path(self, premise_id: str) -> list[Asset]:
        """Equipment from the home up to the feeder head, nearest first."""
        p = self.premises[premise_id]
        path: list[Asset] = []
        cur: str | None = p.transformer
        while cur is not None:
            a = self.assets[cur]
            path.append(a)
            cur = a.parent
        return path

    def feeder_assets(self, feeder: str) -> Iterator[Asset]:
        return (a for a in self.assets.values() if a.feeder == feeder)

    def feeder_premises(self, feeder: str) -> list[Premise]:
        return [p for p in self.premises.values() if p.feeder == feeder]

    def premises_under(self, asset_id: str) -> list[Premise]:
        """Homes whose supply path includes this asset."""
        by_tx: dict[str, list[Premise]] = {}
        for p in self.premises.values():
            by_tx.setdefault(p.transformer, []).append(p)
        out: list[Premise] = []
        stack = [asset_id]
        while stack:
            cur = stack.pop()
            out.extend(by_tx.get(cur, []))
            stack.extend(self.children(cur))
        return out

    def compute_customers_downstream(self) -> None:
        for a in self.assets.values():
            a.customers_downstream = len(self.premises_under(a.id))

    def find_address(self, query: str, limit: int = 5) -> list[Premise]:
        q = query.strip().lower()
        if not q:
            return []
        hits = [p for p in self.premises.values() if q in p.address.lower()]
        hits.sort(key=lambda p: (not p.address.lower().startswith(q), p.address))
        return hits[:limit]

    def nearest_premise(self, x: float, y: float) -> Premise:
        return min(self.premises.values(), key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)

    def summary(self) -> dict:
        by_type: dict[str, int] = {}
        for a in self.assets.values():
            by_type[a.type] = by_type.get(a.type, 0) + 1
        return {
            "name": self.name,
            "feeders": len(self.feeders),
            "assets": len(self.assets),
            "assets_by_type": by_type,
            "premises": len(self.premises),
            "priority_facilities": sum(p.priority for p in self.premises.values()),
        }

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        data = {
            "name": self.name,
            "centre": [self.centre_lat, self.centre_lon],
            "assets": [a.to_dict() for a in self.assets.values()],
            "premises": [p.to_dict() for p in self.premises.values()],
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> Grid:
        data = json.loads(path.read_text())
        g = cls(name=data["name"], centre_lat=data["centre"][0], centre_lon=data["centre"][1])
        for a in data["assets"]:
            g.add(Asset(**a))
        for p in data["premises"]:
            g.premises[p["id"]] = Premise(**p)
        return g


# --- synthetic generator -----------------------------------------------------

_STREETS = ("Maple", "Cedar", "Ridge", "Mill", "Orchard", "Quarry", "Willow", "Hollow", "Summit", "Harbor",
            "Fox Run", "Beacon", "Meadow", "Pine", "Granite", "Juniper", "Elm", "Birch", "Laurel", "Chestnut")
_SUFFIX = ("Rd", "Ln", "Dr", "Ave", "Ct", "Way")


def synthesize(seed: int = 7, feeders: int = 4, laterals_per_feeder: int = 6, poles_per_lateral: int = 7,
               homes_per_transformer: int = 5, name: str = "Harbor Ridge") -> Grid:
    """Generate a plausible radial grid: a substation, several feeders, laterals,
    poles carrying primary conductor, pole-top transformers, and homes.

    Age and vegetation vary spatially (older stock and heavier canopy further
    out) so a fitted model has real structure to find.
    """
    rng = random.Random(seed)
    g = Grid(name=name, centre_lat=39.96, centre_lon=-82.99)
    n_prem = 0
    for f in range(feeders):
        feeder = f"F{f + 1}"
        theta = 2 * math.pi * f / feeders + rng.uniform(-0.2, 0.2)
        head = Asset(id=f"{feeder}-R", type="recloser", feeder=feeder, x=0.0, y=0.0, parent=None,
                     age_years=rng.uniform(2, 25), vegetation=0.1)
        g.add(head)
        prev = head.id
        trunk_veg = rng.uniform(0.2, 0.6)
        # trunk: a run of primary segments between poles
        trunk_poles: list[str] = []
        for s in range(laterals_per_feeder):
            dist = 0.8 * (s + 1) + rng.uniform(-0.1, 0.1)
            x, y = dist * math.cos(theta), dist * math.sin(theta)
            pole = _pole(g, f"{feeder}-P{s}", feeder, x, y, prev, rng, veg=trunk_veg + 0.05 * s, depth=s)
            seg = Asset(id=f"{feeder}-L{s}", type="primary", feeder=feeder, x=x, y=y, parent=pole.id,
                        age_years=pole.age_years, vegetation=min(1.0, trunk_veg + 0.05 * s), length_km=0.8,
                        age_known=pole.age_known)
            g.add(seg)
            prev = seg.id
            trunk_poles.append(seg.id)
        # laterals: a fuse, then poles with transformers, branching off each trunk segment
        for s, trunk_seg in enumerate(trunk_poles):
            side = 1 if s % 2 == 0 else -1
            phi = theta + side * (math.pi / 2 + rng.uniform(-0.3, 0.3))
            base = g.assets[trunk_seg]
            underground = rng.random() < 0.15
            fuse = Asset(id=f"{feeder}-F{s}", type="fuse", feeder=feeder, x=base.x, y=base.y, parent=trunk_seg,
                         age_years=rng.uniform(1, 30), vegetation=base.vegetation)
            g.add(fuse)
            up = fuse.id
            lat_veg = min(1.0, base.vegetation + rng.uniform(0.0, 0.3))
            for k in range(poles_per_lateral):
                d = 0.25 * (k + 1)
                x, y = base.x + d * math.cos(phi), base.y + d * math.sin(phi)
                if underground:
                    seg = Asset(id=f"{feeder}-{s}-U{k}", type="lateral", feeder=feeder, x=x, y=y, parent=up,
                                age_years=rng.uniform(1, 20), vegetation=0.0, underground=True, length_km=0.25)
                    g.add(seg)
                    up = seg.id
                else:
                    pole = _pole(g, f"{feeder}-{s}-P{k}", feeder, x, y, up, rng, veg=lat_veg, depth=s + k)
                    seg = Asset(id=f"{feeder}-{s}-L{k}", type="lateral", feeder=feeder, x=x, y=y, parent=pole.id,
                                age_years=pole.age_years, vegetation=lat_veg, length_km=0.25, age_known=pole.age_known)
                    g.add(seg)
                    up = seg.id
                tx = Asset(id=f"{feeder}-{s}-T{k}", type="transformer", feeder=feeder, x=x, y=y, parent=up,
                           age_years=rng.uniform(1, 35), vegetation=lat_veg * 0.5, underground=underground,
                           kva=rng.choice([25, 37.5, 50, 75]))
                g.add(tx)
                for _ in range(homes_per_transformer):
                    n_prem += 1
                    hx = x + rng.uniform(-0.08, 0.08)
                    hy = y + rng.uniform(-0.08, 0.08)
                    kind = "residential"
                    if rng.random() < 0.06:
                        kind = "commercial"
                    street = f"{rng.choice(_STREETS)} {rng.choice(_SUFFIX)}"
                    g.premises[f"H{n_prem:05d}"] = Premise(
                        id=f"H{n_prem:05d}", address=f"{rng.randint(10, 9900)} {street}", feeder=feeder,
                        transformer=tx.id, x=hx, y=hy, kind=kind)
    # priority facilities: a few homes become hospitals / fire stations / water plants / shelters
    ids = sorted(g.premises)
    for kind, n in (("hospital", 1), ("fire", 2), ("water", 1), ("shelter", 2)):
        for _ in range(n):
            pid = rng.choice(ids)
            g.premises[pid].kind = kind
            g.premises[pid].address = f"{rng.randint(100, 999)} {rng.choice(_STREETS)} {rng.choice(_SUFFIX)} ({kind})"
    g.compute_customers_downstream()
    return g


def _pole(g: Grid, pid: str, feeder: str, x: float, y: float, parent: str, rng: random.Random, *, veg: float,
          depth: int) -> Asset:
    age_known = rng.random() > 0.3  # a realistic share of poles has no install date on record
    age = max(1.0, rng.gauss(22 + 1.5 * depth, 12))
    pole = Asset(id=pid, type="pole", feeder=feeder, x=x, y=y, parent=parent, age_years=age,
                 vegetation=min(1.0, veg), pole_class=rng.choice([2, 3, 4, 4, 5, 5]), age_known=age_known)
    g.add(pole)
    return pole
