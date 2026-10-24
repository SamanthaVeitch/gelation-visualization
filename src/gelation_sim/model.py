"""Shared structural interface for the gelation project.

The public interface in this module follows Project_Roadmap.docx.  Graphical
and statistical modules should depend on this file, not on simulator internals.
Particle IDs are zero-based and immutable after initialization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

SPECIES = ("monomer", "crosslinker", "initiator")


@dataclass(slots=True)
class Particle:
    """Represent one coarse-grained reagent in the simulation.

    ``functionality`` counts remaining unreacted vinyl groups, not graph
    degree.  An incorporated monomer may therefore have functionality zero
    yet carry a radical capable of propagating further.
    """

    id: int
    species: str
    position: np.ndarray
    functionality: int
    active_radical: bool = False
    bonded: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject malformed objects before they enter the shared network."""
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("Particle.id must be a nonnegative integer.")
        if self.species not in SPECIES:
            raise ValueError(f"Unsupported particle species: {self.species}")
        self.position = np.asarray(self.position, dtype=float).copy()
        if self.position.shape != (3,) or not np.all(
                np.isfinite(self.position)):
            raise ValueError("Particle.position must contain three numbers.")
        if not isinstance(self.functionality, int) or self.functionality < 0:
            raise ValueError("Particle.functionality must be nonnegative.")
        if not isinstance(self.active_radical, bool):
            raise ValueError("Particle.active_radical must be Boolean.")
        if not isinstance(self.bonded, list):
            raise ValueError("Particle.bonded must be a list of particle IDs.")


@dataclass(frozen=True, slots=True)
class Bond:
    """Store the two endpoints and the discrete bond formation step."""

    particle_a: int
    particle_b: int
    form_time: int


class Network:
    """Hold the current network topology and coordinates.

    Public API:
        particles: numpy object array of Particle objects, indexed by ID
        bonds: numpy integer array shaped (n_bonds, 2)
        time: discrete simulation step
        seed: random seed used to generate this network

    ``bond_records`` preserves ``Bond.form_time`` without replacing the
    agreed-upon 2-column ``Network.bonds`` array.  All bond modifications must
    pass through ``add_bond`` to keep ``Particle.bonded`` consistent.
    """

    def __init__(
            self,
            particles: list[Particle] | np.ndarray,
            seed: int | None = None,
            time: int = 0) -> None:
        """Create a network containing particles in ascending ID order."""
        self.particles = np.asarray(particles, dtype=object)
        if self.particles.ndim != 1:
            raise ValueError("Network.particles must be a 1-D array.")
        for expected_id, particle in enumerate(self.particles):
            if not isinstance(particle, Particle):
                raise TypeError(
                    "Network.particles must contain Particle objects.")
            if particle.id != expected_id:
                raise ValueError(
                    "Particle IDs must be zero-based and ordered.")
            if particle.bonded:
                raise ValueError("Use Network.add_bond to construct bonds.")
        self.time = time
        self.seed = seed
        self.bond_records: list[Bond] = []
        self._edges: list[tuple[int, int]] = []
        self._edge_set: set[tuple[int, int]] = set()
        self._bonds_cache: np.ndarray | None = None

    @property
    def bonds(self) -> np.ndarray:
        """Return read-only (n, 2) integer bond endpoints for collaborators."""
        if self._bonds_cache is None:
            bonds = np.asarray(self._edges, dtype=np.int64).reshape(-1, 2)
            bonds.flags.writeable = False
            self._bonds_cache = bonds
        return self._bonds_cache

    def add_bond(self, particle_a: int, particle_b: int, step: int) -> Bond:
        """Make one unique undirected edge and update both particle lists."""
        n_particles = len(self.particles)
        if not 0 <= particle_a < n_particles:
            raise ValueError("First bond endpoint is outside the network.")
        if not 0 <= particle_b < n_particles:
            raise ValueError("Second bond endpoint is outside the network.")
        if particle_a == particle_b:
            raise ValueError("Self-bonds are not supported.")
        if not isinstance(step, int) or step < 0:
            raise ValueError("Bond formation step must be nonnegative.")

        a, b = sorted((particle_a, particle_b))
        if (a, b) in self._edge_set:
            raise ValueError(f"Duplicate bond between {a} and {b}.")

        bond = Bond(particle_a=a, particle_b=b, form_time=step)
        self._edges.append((a, b))
        self._edge_set.add((a, b))
        self.particles[a].bonded.append(b)
        self.particles[b].bonded.append(a)
        self.bond_records.append(bond)
        self._bonds_cache = None
        return bond

    def has_bond(self, particle_a: int, particle_b: int) -> bool:
        """Check for an edge in constant expected time."""
        return tuple(sorted((particle_a, particle_b))) in self._edge_set

    def snapshot(self) -> Network:
        """Capture an independent, immutable-in-practice historical state.

        Explicit snapshots cost memory, but only at the selected interval.
        The simulation continues mutating the original Network, never these
        copies.  Callers should treat saved snapshots as read-only.
        """
        return deepcopy(self)
