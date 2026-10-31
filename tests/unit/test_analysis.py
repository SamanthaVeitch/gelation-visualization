"""Tests for gelation analysis helpers."""

import csv
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
    calc_init_eff,
    # initiator
    calc_init,
    calc_half_life,
    calc_init_rate,
    # radicals
    calc_rad,
    calc_rad_ss,
    # polymer
    calc_poly_rate_ss,
    calc_chain_len,
    calc_chain_len_run,
    # data export
    calc_kinetic_params,
    calc_validation_rows,
    calc_validation_constants,
)
from gelation_sim.config import AVOGADRO, SimulationConfig
from gelation_sim.export import export_validation
from gelation_sim.model import Network, Particle
from gelation_sim.simulator import SimulationResult


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


def test_calc_init_eff():
    """Check successful and unsuccessful resolved initiations."""
    events = [
        [1, "initiation", 10, -1],
        [2, "propagation", 10, 20],
        [3, "initiation", 11, -1],
        [4, "termination_disproportionation", 11, 30],
        [5, "initiation", 12, -1],
        [6, "termination_quench", 12, -1],
        [7, "initiation", 13, -1],
    ]

    assert calc_init_eff(events) == pytest.approx(0.5)
    assert calc_init_eff([]) is None


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


def test_calc_half_life():
    """Check effective initiator half-life."""
    assert calc_half_life(0.1) == pytest.approx(
        math.log(2) / 0.1
    )
    assert calc_half_life(None) is None

    with pytest.raises(ValueError):
        calc_half_life(-0.1)
        calc_half_life(0)


def test_calc_init_rate():
    """Check event-based initiation rates and interval boundaries."""
    config = SimulationConfig.from_json("config/example.json")
    steps = [0, 10, 20]

    events = [
        [0, "initiation", 1, -1],
        [5, "initiation", 2, -1],
        [10, "initiation", 3, -1],
        [15, "propagation", 3, 5],
        [20, "initiation", 4, -1],
    ]

    results = calc_init_rate(steps, events, config)

    assert len(results) == 2

    assert results[0][:3] == (0, 10, 2)
    assert results[1][:3] == (10, 20, 1)

    assert results[0][3] == pytest.approx(
        calc_conc(2, config) / calc_interval(0, 10, config)
    )
    assert results[1][3] == pytest.approx(
        calc_conc(1, config) / calc_interval(10, 20, config)
    )

    assert calc_init_rate([0], events, config) == []


# radicals
def test_calc_rad():
    """Check radical concentrations and final quench correction."""
    config = SimulationConfig.from_json("config/example.json")

    snapshots = [
        Network([
            Particle(0, "monomer", (0, 0, 0), 1, active_radical=True),
            Particle(1, "monomer", (1, 0, 0), 1, active_radical=False),
        ], time=0),
        Network([
            Particle(0, "monomer", (0, 0, 0), 1, active_radical=False),
            Particle(1, "monomer", (1, 0, 0), 1, active_radical=False),
        ], time=config.max_steps),
    ]

    events = [
        [config.max_steps, "termination_quench", 0, -1],
    ]

    results = calc_rad(snapshots, events, config)

    assert [row[1] for row in results] == [1, 1]
    assert results[0][2] == pytest.approx(calc_conc(1, config))
    assert results[1][2] == pytest.approx(calc_conc(1, config))


def test_calc_rad_ss():
    """Check theoretical steady-state radical concentration."""
    result = calc_rad_ss(0.01, 0.1, 100.0, 0.5)

    assert result == pytest.approx(
        math.sqrt(0.5 * 0.1 * 0.01 / 100.0)
    )

    assert calc_rad_ss(0, 0.1, 100.0, 0.5) == 0
    assert calc_rad_ss(0.01, 0.1, 0, 0.5) is None
    assert calc_rad_ss(0.01, None, 100.0, 0.5) is None


# polymer
def test_calc_poly_rate_ss():
    """Check theoretical steady-state polymerization rate."""
    assert calc_poly_rate_ss(100.0, 0.01, 0.002) == pytest.approx(
        0.002
    )

    assert calc_poly_rate_ss(100.0, 0, 0.002) == 0
    assert calc_poly_rate_ss(100.0, 0.01, None) is None

    with pytest.raises(ValueError):
        calc_poly_rate_ss(-1.0, 0.01, 0.002)


def test_calc_chain_len():
    """Check aggregate chain length across simulation intervals."""
    events = [
        [1, "initiation", 0, -1],
        [2, "propagation", 0, 1],
        [3, "propagation", 1, 2],
        [4, "termination_combination", 2, 3],
        [5, "propagation", 4, 5],
        [6, "termination_quench", 5, -1],
        [7, "propagation", 6, 7],
        [8, "termination_disproportionation", 7, 8],
    ]

    results = calc_chain_len([0, 5, 10], events)

    assert [row[2:4] for row in results] == [
        (3, 1), (1, 1)
    ]
    assert [row[4] for row in results] == pytest.approx(
        [1.5, 0.5]
    )
    assert calc_chain_len([0], events) == []


def test_calc_chain_len_run():
    """Check completed-run chain length and zero termination."""
    events = [
        [1, "propagation", 0, 1],
        [2, "propagation", 1, 2],
        [3, "termination_combination", 2, 3],
        [4, "propagation", 4, 5],
        [5, "termination_disproportionation", 5, 6],
        [6, "termination_quench", 7, -1],
    ]

    assert calc_chain_len_run(events) == pytest.approx(0.75)
    assert calc_chain_len_run([]) is None


# Data export
def test_calc_validation_rows():
    """Check snapshot row structure and observed concentrations."""
    config = SimulationConfig.from_json("config/example.json")

    snapshots = [
        Network([
            Particle(0, "monomer", (0, 0, 0), 1),
            Particle(1, "monomer", (1, 0, 0), 1),
        ], time=0),
        Network([
            Particle(0, "monomer", (0, 0, 0), 1),
            Particle(1, "monomer", (1, 0, 0), 1),
        ], time=20),
    ]

    result = SimulationResult(
        network=snapshots[-1],
        snapshots=snapshots,
        events_log=[],
        config=config,
    )

    params = calc_kinetic_params(result)
    rows = calc_validation_rows(result, params)

    assert len(rows) == 2
    assert all(len(row) == 14 for row in rows)

    assert rows[0][0] == "step_000000"
    assert rows[1][0] == "step_000020"

    assert rows[1][1] == pytest.approx(calc_time(20, config))
    assert rows[0][3] == pytest.approx(calc_conc(2, config))
    assert rows[0][6:8] == (2, 0)

    assert rows[0][9:12] == (None, None, None)
    assert rows[1][9:12] == (0.0, 0.0, 0.0)
    assert rows[1][12:14] == (None, None)


def test_calc_validation_constants():
    """Check validation constant values and deferred fields."""
    config = SimulationConfig.from_json("config/example.json")

    snapshots = [
        Network([
            Particle(0, "monomer", (0, 0, 0), 1),
            Particle(1, "monomer", (1, 0, 0), 1),
        ], time=0),
        Network([
            Particle(0, "monomer", (0, 0, 0), 1),
            Particle(1, "monomer", (1, 0, 0), 1),
        ], time=20),
    ]

    result = SimulationResult(
        network=snapshots[-1],
        snapshots=snapshots,
        events_log=[],
        config=config,
    )

    params = calc_kinetic_params(result)
    constants = calc_validation_constants(result, params)
    values = {name: value for name, value, units in constants}

    assert len(constants) == 30
    assert values["Seed"] == config.seed
    assert values["Box length"] == config.box_length_nm
    assert values["Total time"] == pytest.approx(
        calc_time(20, config)
    )
    assert values["Initiator efficiency"] is None
    assert values["Predicted mean kinetic chain length"] is None
    assert values["Mn"] is None
    assert values["Mw"] is None
    assert values["PDI"] is None


# CSV
def test_export_validation(tmp_path):
    """Check validation CSV headers, values, and empty fields."""
    rows = [
        (
            "step_000000", 0.0, 0.01, 0.2, 0.0,
            10, 200, 0, 1.0, None, None, None,
            None, None,
        ),
    ]
    constants = [
        ("Seed", 42, None),
        ("kd", 0.001, "1/s"),
        ("Mn", None, "g/mol"),
    ]

    constants_path, equations_path = export_validation(
        rows, constants, tmp_path
    )

    with constants_path.open(newline="", encoding="utf-8") as handle:
        constant_data = list(csv.reader(handle))

    with equations_path.open(newline="", encoding="utf-8") as handle:
        equation_data = list(csv.reader(handle))

    assert constant_data == [
        ["Value", "", "Units"],
        ["Seed", "42", ""],
        ["kd", "0.001", "1/s"],
        ["Mn", "", "g/mol"],
    ]

    assert len(equation_data[0]) == 14
    assert equation_data[0][0] == ""
    assert equation_data[0][11] == (
        "Simulated polymerization rate (mol/(L*s))"
    )
    assert equation_data[0][12].startswith("Theoretical")
    assert equation_data[0][13].startswith("Theoretical")

    assert equation_data[1][0] == "step_000000"
    assert equation_data[1][1] == "0.0"
    assert equation_data[1][9:] == ["", "", "", "", ""]
