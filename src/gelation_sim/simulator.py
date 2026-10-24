"""Brownian encounter / free-radical polymer network formation engine.

Mechanism:
1. Place molecular nodes uniformly in a cubic periodic box.
2. Displace nodes diffusively and enforce approximate bond lengths without
   an atomistic force field.
3. Activate initiators with a Bernoulli trial per unactivated molecule.
4. Resolve radical-radical termination for nearby active sites.
5. Propagate surviving radicals to nearby unreacted vinyl groups.
6. Record reaction events and occasional independent Network snapshots.

The model is coarse-grained.  In particular, one crosslinker node has many
vinyl groups but at most one active radical because the roadmap defines a
Boolean rather than a radical count.  This simplification is intentional and
limits quantitative interpretation of reaction kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import SimulationConfig
from .model import Network, Particle


@dataclass(slots=True)
class SimulationResult:
    """Bundle final network, independent snapshots, and chemical events."""

    network: Network
    snapshots: list[Network]
    events_log: list[list[int | str]]
    config: SimulationConfig


class Simulator:
    """Simulate free-radical gelation in a periodic, cubic solution volume."""

    def __init__(
            self,
            config: SimulationConfig,
            network: Network | None = None) -> None:
        """Initialize a seeded RNG and the initial network.

        ``network`` can be injected for deterministic mechanism tests.  Its
        particles must satisfy the normal roadmap interface and be inside
        the configured box; seed remains specified by ``config``.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.network = network if network is not None else self._make_network()
        if self.network.time != 0:
            raise ValueError(
                "Simulator requires an initial network at time 0.")
        box = config.box_length_nm
        if any(np.any(p.position < 0) or np.any(p.position >= box)
               for p in self.network.particles):
            raise ValueError(
                "Initial particle coordinates must be in the box.")
        self.snapshots: list[Network] = []
        self.events_log: list[list[int | str]] = []
        self._activated_initiators: set[int] = {
            p.id for p in self.network.particles
            if p.species == "initiator" and p.active_radical
        }
        self._has_run = False

    def _make_network(self) -> Network:
        """Sample independent initial 3-D positions for each reagent."""
        particles: list[Particle] = []
        counts = self.config.particle_counts()
        for species in ("monomer", "crosslinker", "initiator"):
            for _ in range(counts[species]):
                functionality = {
                    "monomer": 1,
                    "crosslinker": self.config.crosslinker_functionality,
                    "initiator": 0,
                }[species]
                particles.append(Particle(
                    id=len(particles),
                    species=species,
                    position=self.rng.uniform(
                        0, self.config.box_length_nm, size=3),
                    functionality=functionality,
                ))
        return Network(particles=particles, seed=self.config.seed)

    def _move_particles(self) -> np.ndarray:
        """Diffuse each node, then geometrically relax existing bonds.

        A 3-D Brownian increment has independent coordinate variance
        ``2 * D * dt``.  Periodic wrapping prevents loss of molecules at the
        boundary.  Bond-length projection is a geometric constraint only;
        there is no force field, hydrodynamics, or excluded-volume potential.
        """
        particles = self.network.particles
        if len(particles) == 0:
            return np.empty((0, 3), dtype=float)

        positions = np.stack([p.position for p in particles])
        diffusion = {
            "monomer": self.config.monomer_diffusion_nm2_s,
            "crosslinker": self.config.crosslinker_diffusion_nm2_s,
            "initiator": self.config.initiator_diffusion_nm2_s,
        }
        widths = np.array([
            np.sqrt(2 * diffusion[p.species] * self.config.time_step_s)
            for p in particles
        ])
        positions += self.rng.normal(size=positions.shape) * widths[:, None]
        positions %= self.config.box_length_nm

        bonds = self.network.bonds
        if len(bonds) > 0 and self.config.constraint_iterations > 0:
            a_ids = bonds[:, 0]
            b_ids = bonds[:, 1]
            degrees = np.bincount(
                bonds.flatten(), minlength=len(particles)).clip(min=1)
            box = self.config.box_length_nm
            for _ in range(self.config.constraint_iterations):
                displacement = positions[b_ids] - positions[a_ids]
                displacement = (displacement + box / 2) % box - box / 2
                distances = np.linalg.norm(displacement, axis=1)
                # A zero-length edge has no defined direction; another
                # Brownian displacement can resolve it in a later step.
                scale = np.divide(
                    0.5 * (distances - self.config.bond_length_nm),
                    distances,
                    out=np.zeros_like(distances),
                    where=distances > 1.0e-12,
                )
                corrections = displacement * scale[:, None]
                increments = np.zeros_like(positions)
                np.add.at(increments, a_ids, corrections)
                np.add.at(increments, b_ids, -corrections)
                positions = (positions
                             + increments / degrees[:, None]) % box

        # Preserve the same Particle object instances used by collaborators.
        for particle, position in zip(particles, positions):
            particle.position[:] = position
        return positions

    def _activate_initiators(self, step: int) -> None:
        """Turn a previously unused initiator into a single active radical."""
        p_activate = self.config.initiation_probability
        for particle in self.network.particles:
            if particle.species != "initiator":
                continue
            if particle.id in self._activated_initiators:
                continue
            if self.rng.random() < p_activate:
                particle.active_radical = True
                self._activated_initiators.add(particle.id)
                self.events_log.append([
                    step, "initiation", particle.id, -1,
                ])

    def _terminate_radicals(self, step: int, tree: cKDTree) -> None:
        """Choose combination or disproportionation for adjacent radicals.

        One unordered radical pair receives at most one categorical trial.
        Combination adds a bond; disproportionation ends both radicals
        without forming a bond.  Occupied edges cannot receive a duplicate
        combination bond, but may undergo disproportionation.
        """
        config = self.config
        radicals = [p.id for p in self.network.particles
                    if p.active_radical]
        if len(radicals) < 2:
            return
        self.rng.shuffle(radicals)
        visited: set[int] = set()
        for particle_id in radicals:
            source = self.network.particles[particle_id]
            if not source.active_radical or particle_id in visited:
                continue
            neighbors = tree.query_ball_point(
                source.position, r=config.capture_radius_nm)
            neighbors = [
                neighbor for neighbor in neighbors
                if neighbor != particle_id
                and self.network.particles[neighbor].active_radical
                and neighbor not in visited
            ]
            self.rng.shuffle(neighbors)
            for target_id in neighbors:
                target = self.network.particles[target_id]
                if not target.active_radical:
                    continue
                chance = self.rng.random()
                if chance < config.termination_combination_probability:
                    if self.network.has_bond(particle_id, target_id):
                        continue
                    self.network.add_bond(particle_id, target_id, step)
                    event = "termination_combination"
                elif chance < (
                        config.termination_combination_probability
                        + config.termination_disproportionation_probability):
                    event = "termination_disproportionation"
                else:
                    continue
                source.active_radical = False
                target.active_radical = False
                self.events_log.append([
                    step, event, particle_id, target_id,
                ])
                visited.add(particle_id)
                visited.add(target_id)
                break

    def _propagate_radicals(self, step: int, tree: cKDTree) -> None:
        """Transfer each active radical to one neighboring vinyl reagent.

        A propagating radical is consumed on the source and created on the
        target.  The target loses precisely one available vinyl group.  A
        multifunctional target gets a higher per-encounter probability,
        ``1 - (1-p)^f``, representing f independent accessible sites.
        """
        radicals = [p.id for p in self.network.particles
                    if p.active_radical]
        self.rng.shuffle(radicals)
        for particle_id in radicals:
            source = self.network.particles[particle_id]
            if not source.active_radical:
                continue
            neighbors = tree.query_ball_point(
                source.position, r=self.config.capture_radius_nm)
            self.rng.shuffle(neighbors)
            for target_id in neighbors:
                if target_id == particle_id:
                    continue
                target = self.network.particles[target_id]
                if target.functionality == 0 or target.active_radical:
                    continue
                if self.network.has_bond(particle_id, target_id):
                    continue
                probability = 1 - (
                    1 - self.config.propagation_probability
                ) ** target.functionality
                if self.rng.random() >= probability:
                    continue
                self.network.add_bond(particle_id, target_id, step)
                source.active_radical = False
                target.active_radical = True
                target.functionality -= 1
                self.events_log.append([
                    step, "propagation", particle_id, target_id,
                ])
                break

    def _quench_radicals(self, step: int) -> None:
        """Optionally end unfinished chains when the finite run is stopped.

        This is an explicit end-of-simulation assumption, not a spontaneous
        termination rate inferred from radical chemistry.
        """
        for particle in self.network.particles:
            if particle.active_radical:
                particle.active_radical = False
                self.events_log.append([
                    step, "termination_quench", particle.id, -1,
                ])

    def run(self) -> SimulationResult:
        """Execute one simulation and return complete reproducible outputs."""
        if self._has_run:
            raise RuntimeError("Simulator.run may only be invoked once.")
        self._has_run = True
        self.snapshots.append(self.network.snapshot())

        for step in range(1, self.config.max_steps + 1):
            positions = self._move_particles()
            self._activate_initiators(step)
            if len(positions) > 0:
                tree = cKDTree(
                    positions, boxsize=self.config.box_length_nm)
                self._terminate_radicals(step, tree)
                self._propagate_radicals(step, tree)
            self.network.time = step
            if step % self.config.snapshot_interval == 0:
                self.snapshots.append(self.network.snapshot())

        if self.config.quench_at_end:
            self._quench_radicals(self.config.max_steps)

        # Replace the last snapshot when the quench occurs on the same step.
        # This ensures the final snapshot always equals the returned network.
        if self.snapshots[-1].time == self.network.time:
            self.snapshots[-1] = self.network.snapshot()
        else:
            self.snapshots.append(self.network.snapshot())
        return SimulationResult(
            network=self.network,
            snapshots=self.snapshots,
            events_log=self.events_log,
            config=self.config,
        )
