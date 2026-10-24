"""Stable CSV exchange schema tests."""

import csv
import json

from gelation_sim import Network, Particle, SimulationConfig, Simulator
from gelation_sim.export import export_result


def test_export_is_machine_readable_with_complete_snapshots(tmp_path):
    config = SimulationConfig(
        box_length_nm=6,
        max_steps=3,
        snapshot_interval=2,
        seed=4,
    )
    result = Simulator(config).run()
    root = export_result(result, tmp_path / "run")
    assert (root / "network_particles.csv").exists()
    assert (root / "network_bonds.csv").exists()
    assert (root / "events.csv").exists()
    assert (root / "snapshot_index.csv").exists()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.0"
    assert manifest["particle_count"] == len(result.network.particles)
    assert manifest["final_step"] == 3
    assert len(list((root / "snapshots").glob("*_particles.csv"))) == 3

    with (root / "network_particles.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.network.particles)
    assert set(rows[0]) == {
        "id", "species", "x_nm", "y_nm", "z_nm", "functionality",
        "active_radical", "bonded",
    }

    with (root / "snapshot_index.csv").open(newline="") as handle:
        index = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in index] == [0, 2, 3]
    for row in index:
        assert (root / row["particles_file"]).exists()
        assert (root / row["bonds_file"]).exists()


def test_export_preserves_bond_time_and_empty_event_header(tmp_path):
    config = SimulationConfig(max_steps=1)
    network = Network([
        Particle(0, "monomer", [1, 1, 1], 1),
        Particle(1, "crosslinker", [2, 1, 1], 2),
    ])
    network.add_bond(0, 1, step=0)
    # Use the normal simulation run, with probabilities disabled, so the
    # final output has a real edge but no chemical events to export.
    from dataclasses import replace
    config = replace(
        config,
        initiation_probability=0,
        propagation_probability=0,
        termination_combination_probability=0,
        termination_disproportionation_probability=0,
    )
    # The simulator test initial Network only needs its bonded state.
    result = Simulator(config, network=network).run()
    output = export_result(result, tmp_path / "initial_bond")
    with (output / "network_bonds.csv").open(newline="") as handle:
        bonds = list(csv.DictReader(handle))
    assert len(bonds) == 1
    assert bonds[0]["form_time"] == "0"
    assert (output / "events.csv").read_text().splitlines() == [
        "step,reaction,particle_a,particle_b",
    ]


def test_roundtrip_result_recovers_complete_structures(tmp_path):
    """Collaborators can load a full run without accessing simulator code."""
    import numpy as np

    from gelation_sim import load_result

    config = SimulationConfig(
        max_steps=7,
        snapshot_interval=3,
        box_length_nm=8,
        seed=19,
    )
    original = Simulator(config).run()
    path = export_result(original, tmp_path / "roundtrip")
    restored = load_result(path)
    assert restored.config == original.config
    assert restored.events_log == original.events_log
    assert restored.network.time == original.network.time
    assert np.array_equal(restored.network.bonds, original.network.bonds)
    assert [snapshot.time for snapshot in restored.snapshots] == [0, 3, 6, 7]
    for before, after in zip(
            original.network.particles, restored.network.particles):
        assert before.id == after.id
        assert before.species == after.species
        assert before.bonded == after.bonded
        assert before.functionality == after.functionality
        assert before.active_radical == after.active_radical
        np.testing.assert_array_equal(before.position, after.position)


def test_overwriting_output_removes_outdated_generated_snapshots(tmp_path):
    config = SimulationConfig(max_steps=6, snapshot_interval=1, seed=2)
    root = tmp_path / "same_run"
    export_result(Simulator(config).run(), root)
    assert (root / "snapshots" / "step_000005_particles.csv").exists()
    from dataclasses import replace

    next_config = replace(config, max_steps=2, snapshot_interval=2)
    export_result(Simulator(next_config).run(), root)
    assert not (root / "snapshots" / "step_000005_particles.csv").exists()
    assert len(list((root / "snapshots").glob("*_particles.csv"))) == 2
