"""Contracts required by visualization and analysis collaborators."""

import numpy as np
import pytest

from gelation_sim import Bond, Network, Particle


def test_empty_network_has_two_column_integer_bonds():
    network = Network([], seed=7)
    assert isinstance(network.particles, np.ndarray)
    assert network.particles.shape == (0,)
    assert isinstance(network.bonds, np.ndarray)
    assert network.bonds.shape == (0, 2)
    assert np.issubdtype(network.bonds.dtype, np.integer)
    assert network.seed == 7


def test_add_bond_maintains_two_adjacency_lists_and_timestamp():
    particles = [
        Particle(0, "monomer", [0.0, 0.0, 0.0], 1),
        Particle(1, "crosslinker", [1.0, 0.0, 0.0], 2),
    ]
    network = Network(particles)
    assert network.bonds.shape == (0, 2)
    bond = network.add_bond(1, 0, step=3)
    assert bond == Bond(0, 1, 3)
    assert np.array_equal(network.bonds, [[0, 1]])
    assert particles[0].bonded == [1]
    assert particles[1].bonded == [0]
    assert network.has_bond(0, 1)
    assert network.has_bond(1, 0)
    with pytest.raises(ValueError, match="Duplicate bond"):
        network.add_bond(0, 1, step=4)
    with pytest.raises(ValueError, match="Self-bonds"):
        network.add_bond(0, 0, step=4)
    with pytest.raises(ValueError, match="outside"):
        network.add_bond(0, 2, step=4)


def test_snapshot_is_independent_of_future_mutations():
    particles = [
        Particle(0, "monomer", [0.0, 0.0, 0.0], 1),
        Particle(1, "monomer", [1.0, 0.0, 0.0], 1),
    ]
    network = Network(particles, seed=3)
    first = network.snapshot()
    network.add_bond(0, 1, step=1)
    network.time = 1
    network.particles[0].position[0] = 4.0
    network.particles[1].functionality = 0
    assert first.bonds.shape == (0, 2)
    assert first.time == 0
    assert first.particles[0].position[0] == 0.0
    assert first.particles[1].functionality == 1
    assert first.particles[0] is not network.particles[0]


def test_particle_validation():
    with pytest.raises(ValueError, match="three"):
        Particle(0, "monomer", [0.0, 1.0], 1)
    with pytest.raises(ValueError, match="functionality"):
        Particle(0, "monomer", [0, 0, 0], -1)
    with pytest.raises(ValueError, match="species"):
        Particle(0, "solvent", [0, 0, 0], 0)
    with pytest.raises(ValueError, match="zero-based"):
        Network([Particle(1, "monomer", [0, 0, 0], 1)])
