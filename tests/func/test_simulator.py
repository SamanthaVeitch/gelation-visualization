"""Reaction, diffusion, periodic geometry, and reproducibility tests."""

from dataclasses import replace

import numpy as np
import pytest

from gelation_sim import Network, Particle, SimulationConfig, Simulator


def sparse_config(**overrides):
    """Return a small deterministic model for two-particle reaction tests."""
    options = dict(
        monomer_concentration_mol_l=0,
        crosslinker_concentration_mol_l=0,
        initiator_concentration_mol_l=0,
        box_length_nm=5,
        max_steps=1,
        snapshot_interval=1,
        capture_radius_nm=1.0,
        bond_length_nm=0.8,
        monomer_diffusion_nm2_s=0,
        crosslinker_diffusion_nm2_s=0,
        initiator_diffusion_nm2_s=0,
        termination_combination_probability=0,
        termination_disproportionation_probability=0,
        initiation_probability=1,
        propagation_probability=1,
        quench_at_end=False,
        seed=1,
    )
    options.update(overrides)
    return SimulationConfig(**options)


def test_radical_initiation_and_propagation_consume_vinyl():
    config = sparse_config()
    network = Network([
        Particle(0, "initiator", [2, 2, 2], 0),
        Particle(1, "monomer", [2.5, 2, 2], 1),
    ], seed=1)
    result = Simulator(config, network=network).run()
    assert np.array_equal(result.network.bonds, [[0, 1]])
    assert result.network.particles[0].functionality == 0
    assert result.network.particles[1].functionality == 0
    assert not result.network.particles[0].active_radical
    assert result.network.particles[1].active_radical
    assert result.events_log == [
        [1, "initiation", 0, -1],
        [1, "propagation", 0, 1],
    ]
    assert result.snapshots[0].time == 0
    assert result.snapshots[0].bonds.shape == (0, 2)
    assert result.snapshots[-1].time == 1


def test_periodic_boundary_encounter():
    config = sparse_config(capture_radius_nm=0.5, bond_length_nm=0.25)
    network = Network([
        Particle(0, "initiator", [0.1, 2, 2], 0),
        Particle(1, "monomer", [4.9, 2, 2], 1),
    ])
    result = Simulator(config, network).run()
    assert len(result.network.bond_records) == 1


def test_combination_adds_bond_and_disproportionation_does_not():
    particles = [
        Particle(0, "monomer", [1, 1, 1], 0, active_radical=True),
        Particle(1, "monomer", [1.5, 1, 1], 0, active_radical=True),
    ]
    config = sparse_config(
        termination_combination_probability=1,
        propagation_probability=0,
    )
    combination = Simulator(config, Network(particles)).run()
    assert combination.network.bonds.shape == (1, 2)
    assert {event[1] for event in combination.events_log} == {
        "termination_combination"}
    assert all(not p.active_radical for p in combination.network.particles)

    particles = [
        Particle(0, "monomer", [1, 1, 1], 0, active_radical=True),
        Particle(1, "monomer", [1.5, 1, 1], 0, active_radical=True),
    ]
    config = replace(
        config,
        termination_combination_probability=0,
        termination_disproportionation_probability=1,
    )
    disproportionation = Simulator(config, Network(particles)).run()
    assert disproportionation.network.bonds.shape == (0, 2)
    assert disproportionation.events_log[0][1] == (
        "termination_disproportionation")
    assert all(not p.active_radical
               for p in disproportionation.network.particles)


def test_final_quench_and_last_snapshot_are_consistent():
    config = sparse_config(quench_at_end=True)
    network = Network([
        Particle(0, "initiator", [2, 2, 2], 0),
    ])
    result = Simulator(config, network).run()
    assert [event[1] for event in result.events_log] == [
        "initiation", "termination_quench",
    ]
    assert not result.network.particles[0].active_radical
    assert not result.snapshots[-1].particles[0].active_radical


def test_snapshots_include_final_step_even_off_interval():
    config = sparse_config(max_steps=5, snapshot_interval=2)
    result = Simulator(config).run()
    assert [snapshot.time for snapshot in result.snapshots] == [
        0, 2, 4, 5,
    ]
    assert result.network.bonds.shape == (0, 2)


def test_fixed_seed_reproduces_events_positions_and_edges():
    config = sparse_config(
        monomer_concentration_mol_l=0.4,
        crosslinker_concentration_mol_l=0.09,
        initiator_concentration_mol_l=0.12,
        max_steps=12,
        snapshot_interval=5,
        propagation_probability=0.6,
        termination_combination_probability=0.1,
        quench_at_end=True,
    )
    first = Simulator(config).run()
    second = Simulator(config).run()
    assert first.events_log == second.events_log
    assert np.array_equal(first.network.bonds, second.network.bonds)
    coords_a = [particle.position for particle in first.network.particles]
    coords_b = [particle.position for particle in second.network.particles]
    np.testing.assert_array_equal(coords_a, coords_b)
    assert [snapshot.time for snapshot in first.snapshots] == [
        0, 5, 10, 12,
    ]
    simulator = Simulator(config)
    simulator.run()
    with pytest.raises(RuntimeError, match="only be invoked once"):
        simulator.run()


def test_no_reactions_when_all_probabilities_zero():
    config = sparse_config(
        monomer_concentration_mol_l=0.4,
        initiator_concentration_mol_l=0.3,
        initiation_probability=0,
        propagation_probability=0,
    )
    result = Simulator(config).run()
    assert result.events_log == []
    assert len(result.network.bond_records) == 0


def test_mass_balance_bond_balance_and_adjacency_invariants():
    """Each vinyl attack consumes exactly one site and creates one edge."""
    config = sparse_config(
        monomer_concentration_mol_l=0.6,
        crosslinker_concentration_mol_l=0.15,
        initiator_concentration_mol_l=0.06,
        crosslinker_functionality=3,
        max_steps=30,
        snapshot_interval=10,
        capture_radius_nm=1.5,
        seed=8,
        termination_combination_probability=0.08,
        termination_disproportionation_probability=0.02,
        propagation_probability=0.6,
    )
    result = Simulator(config).run()
    initial_vinyl = sum(
        1 if p.species == "monomer"
        else config.crosslinker_functionality
        if p.species == "crosslinker" else 0
        for p in result.network.particles
    )
    remaining_vinyl = sum(p.functionality for p in result.network.particles)
    propagation = sum(
        event[1] == "propagation" for event in result.events_log
    )
    combination = sum(
        event[1] == "termination_combination"
        for event in result.events_log
    )
    assert initial_vinyl - remaining_vinyl == propagation
    assert len(result.network.bond_records) == propagation + combination
    assert len(result.network.bond_records) == len(result.network.bonds)
    assert np.all(result.network.bonds[:, 0]
                  < result.network.bonds[:, 1])
    for particle in result.network.particles:
        assert all(
            particle.id in result.network.particles[neighbor].bonded
            for neighbor in particle.bonded
        )
        assert np.all(particle.position >= 0)
        assert np.all(particle.position < config.box_length_nm)
    assert all(0 <= bond.form_time <= config.max_steps
               for bond in result.network.bond_records)
