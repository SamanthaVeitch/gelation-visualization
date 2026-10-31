"""Tests for gelation analysis helpers."""

import pytest

from gelation_sim.analysis import (
    calc_conc,
    calc_interval,
    calc_time,
    calc_init,
)
from gelation_sim.config import AVOGADRO, SimulationConfig


@pytest.fixture
def config():
    """Load the example simulation configuration."""
    return SimulationConfig.from_json("config/example.json")


def test_calc_conc(config):
    """Check concentration using the configured box volume."""
    vol = config.box_length_nm ** 3 * 1.0e-24

    assert calc_conc(0, config) == 0
    assert calc_conc(100, config) == pytest.approx(
        100 / (AVOGADRO * vol)
    )


def test_calc_time(config):
    """Check physical time using the configured timestep."""
    assert calc_time(20, config) == pytest.approx(
        20 * config.time_step_s
    )


def test_calc_interval(config):
    """Check elapsed time between two simulation steps."""
    assert calc_interval(20, 40, config) == pytest.approx(
        20 * config.time_step_s
    )


def test_calc_init():
    """Check initiator depletion at multiple and individual steps."""
    config = SimulationConfig.from_json("config/example.json")
    events = [
        [1, "initiation", 10, -1],
        [2, "propagation", 10, 20],
        [5, "initiation", 11, -1],
    ]

    initial = config.particle_counts()["initiator"]
    results = calc_init([0, 3, 5], events, config)

    assert [row[1] for row in results] == [
        initial, initial - 1, initial - 2
    ]
    assert results[0][2] == pytest.approx(1.0)
    assert results[1][3] == pytest.approx(
        calc_conc(initial - 1, config)
    )

    single = calc_init([3], events, config)
    assert single == [results[1]]
