import pytest

from lantern.engine import Engine


@pytest.fixture(scope="session")
def engine() -> Engine:
    return Engine.demo(seed=7)


@pytest.fixture(scope="session")
def premise_id(engine) -> str:
    return engine.lookup("Maple")["matches"][0]["premise_id"]
