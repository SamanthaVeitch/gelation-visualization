
"""Functional test for the complete validation export workflow."""

import csv
from dataclasses import replace

import pytest

from gelation_sim import (
    SimulationConfig,
    Simulator,
    export_result,
    load_result,
)
from gelation_sim.analysis import (
    calc_conc,
    calc_kinetic_params,
    calc_time,
    calc_validation_constants,
    calc_validation_rows,
)
from gelation_sim.export import (
    VALIDATION_EQUATION_FIELDS,
    export_validation,
)


def test_validation_integration(tmp_path):
    """Verify validation CSVs generated from a reloaded simulation."""
    config = replace(
        SimulationConfig.from_json("config/example.json"),
        max_steps=8,
        snapshot_interval=2,
    )

    result = Simulator(config).run()
    output_dir = tmp_path / "run"
    export_result(result, output_dir)

    loaded = load_result(output_dir)

    params = calc_kinetic_params(loaded)
    rows = calc_validation_rows(loaded, params)
    constants = calc_validation_constants(loaded, params)

    constants_path, equations_path = export_validation(
        rows, constants, output_dir
    )

    with constants_path.open(
        newline="", encoding="utf-8"
    ) as handle:
        constant_data = list(csv.reader(handle))

    with equations_path.open(
        newline="", encoding="utf-8"
    ) as handle:
        equation_data = list(csv.reader(handle))

    assert constant_data[0] == ["Value", "", "Units"]
    assert len(constant_data) == 31

    assert equation_data[0] == list(VALIDATION_EQUATION_FIELDS)
    assert len(equation_data) == len(loaded.snapshots) + 1
    assert all(len(row) == 14 for row in equation_data)

    values = {
        name: value
        for name, value, units in constant_data[1:]
    }

    assert int(values["Seed"]) == config.seed
    assert float(values["Box length"]) == config.box_length_nm
    assert float(values["Total time"]) == pytest.approx(
        calc_time(config.max_steps, config)
    )

    assert values["Mn"] == ""
    assert values["Mw"] == ""
    assert values["PDI"] == ""

    for i, snapshot in enumerate(loaded.snapshots):
        row = equation_data[i + 1]
        step = snapshot.time

        vinyl_num = sum(
            particle.functionality
            for particle in snapshot.particles
        )

        assert row[0] == f"step_{step:06d}"
        assert float(row[1]) == pytest.approx(
            calc_time(step, config)
        )
        assert float(row[3]) == pytest.approx(
            calc_conc(vinyl_num, config)
        )
        assert int(row[6]) == vinyl_num
        assert int(row[7]) == params["rad_data"][i][1]

    assert equation_data[1][9:12] == ["", "", ""]

    for i in range(1, len(loaded.snapshots)):
        start = loaded.snapshots[i - 1].time
        end = loaded.snapshots[i].time

        init_num = sum(
            1 for step, reaction, a, b in loaded.events_log
            if start < step <= end and reaction == "initiation"
        )

        dt = calc_time(end - start, config)
        expected_rate = calc_conc(init_num, config) / dt

        assert float(equation_data[i + 1][9]) == pytest.approx(
            expected_rate
        )
