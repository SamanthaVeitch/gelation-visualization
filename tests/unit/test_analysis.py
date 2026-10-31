"""Tests for gelation analysis helpers."""

import math
import pytest

from gelation_sim.analysis import (
    # basic helpers
    calc_conc,
    calc_interval,
    calc_time,
    # constants
    calc_kd,
    calc_kp,
    fit_kp,
    calc_kt,
    fit_kt,
    # initiator
    calc_init
)
from gelation_sim.config import AVOGADRO, SimulationConfig
from gelation_sim.model import Network, Particle


@pytest.fixture
def config():
    """Load the example simulation configuration."""
    return SimulationConfig.from_json("config/example.json")


# basic helpers
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


# constants
def test_calc_kd():
    """Check kd estimated from an exponential depletion curve."""
    config = SimulationConfig.from_json("config/example.json")
    kd = 0.1
    dt = config.time_step_s

    init_data = [
        (step, 100, math.exp(-kd * step * dt), 0.0)
        for step in [0, 20, 40, 60]
    ]

    assert calc_kd(init_data, config) == pytest.approx(kd)


def test_calc_kp():
    """Check propagation rate, concentration product, and effective kp."""
    config = SimulationConfig.from_json("config/example.json")

    particles = [
        Particle(0, "initiator", [0, 0, 0], 0, True),
        Particle(1, "monomer", [1, 0, 0], 1),
    ]
    start = Network(particles, time=0)

    particles_end = [
        Particle(0, "initiator", [0, 0, 0], 0),
        Particle(1, "monomer", [1, 0, 0], 0, True),
    ]
    end = Network(particles_end, time=20)

    events = [[10, "propagation", 0, 1]]
    results = calc_kp([start, end], events, config)

    vol = config.box_length_nm ** 3 * 1.0e-24
    scale = AVOGADRO * vol

    expected_rp = 1 / scale
    expected_product = 1 / (2 * scale ** 2)
    expected_kp = 2 * scale

    assert len(results) == 1
    assert results[0][0:3] == (0, 20, 1)
    assert results[0][3] == pytest.approx(expected_rp)
    assert results[0][4] == pytest.approx(expected_product)
    assert results[0][5] == pytest.approx(expected_kp)


def test_fit_kp():
    """Check least-squares fitting of propagation rates."""
    kp_data = [
        (0, 20, 2, 4.0, 2.0, 2.0),
        (20, 40, 3, 6.0, 3.0, 2.0),
        (40, 60, 0, 0.0, 1.0, 0.0),
    ]

    expected = (2.0 * 4.0 + 3.0 * 6.0) / (
        2.0 ** 2 + 3.0 ** 2 + 1.0 ** 2
    )

    assert fit_kp(kp_data) == pytest.approx(expected)


def test_calc_kt():
    """Check termination rate and effective kt."""
    config = SimulationConfig.from_json("config/example.json")

    particles = [
        Particle(0, "initiator", [0, 0, 0], 0, True),
        Particle(1, "initiator", [1, 0, 0], 0, True),
    ]
    start = Network(particles, time=0)

    particles_end = [
        Particle(0, "initiator", [0, 0, 0], 0),
        Particle(1, "initiator", [1, 0, 0], 0),
    ]
    end = Network(particles_end, time=20)

    events = [[10, "termination_disproportionation", 0, 1]]
    results = calc_kt([start, end], events, config)

    vol = config.box_length_nm ** 3 * 1.0e-24
    scale = AVOGADRO * vol

    expected_rt = 2 / scale
    expected_squared = 2 / scale ** 2
    expected_kt = scale / 2

    assert len(results) == 1
    assert results[0][0:3] == (0, 20, 1)
    assert results[0][3] == pytest.approx(expected_rt)
    assert results[0][4] == pytest.approx(expected_squared)
    assert results[0][5] == pytest.approx(expected_kt)
    assert results[0][6] is None


def test_fit_kt():
    """Check termination fitting with a conversion cutoff."""
    kt_data = [
        (0, 20, 2, 8.0, 2.0, 2.0, 0.02),
        (20, 40, 3, 12.0, 3.0, 2.0, 0.08),
        (40, 60, 0, 0.0, 1.0, 0.0, 0.15),
    ]

    expected = (4.0 * 8.0 + 6.0 * 12.0) / (
        4.0 ** 2 + 6.0 ** 2
    )

    assert fit_kt(kt_data) == pytest.approx(expected)
    assert fit_kt(kt_data, max_conv=0.01) is None


# initiator
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
