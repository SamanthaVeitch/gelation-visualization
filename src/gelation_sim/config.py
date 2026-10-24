"""Validated, reproducible inputs for coarse-grained radical gelation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import math

AVOGADRO = 6.02214076e23
NM3_TO_LITERS = 1.0e-24


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Define model geometry, reagent concentrations, and reaction rules.

    Concentrations are mol/L.  Spatial dimensions are nm; diffusion
    coefficients are nm^2/s; ``time_step_s`` is seconds.  Time-dependent
    reaction *probabilities* are probabilities per eligible encounter per
    simulation step, not experimentally calibrated kinetic rate constants.

    One particle is one coarse-grained monomer, multifunctional crosslinker,
    or initiator molecule.  Counts follow C * volume * Avogadro's number,
    rounded to the nearest integer.
    """

    monomer_concentration_mol_l: float = 0.40
    crosslinker_concentration_mol_l: float = 0.05
    initiator_concentration_mol_l: float = 0.012
    crosslinker_functionality: int = 2
    box_length_nm: float = 12.0
    seed: int = 42
    max_steps: int = 300
    snapshot_interval: int = 20
    time_step_s: float = 0.05
    capture_radius_nm: float = 1.50
    bond_length_nm: float = 0.82
    constraint_iterations: int = 8
    monomer_diffusion_nm2_s: float = 0.8
    crosslinker_diffusion_nm2_s: float = 0.5
    initiator_diffusion_nm2_s: float = 1.0
    initiation_probability: float = 0.12
    propagation_probability: float = 0.45
    termination_combination_probability: float = 0.03
    termination_disproportionation_probability: float = 0.01
    quench_at_end: bool = True

    def __post_init__(self) -> None:
        """Fail early for invalid physical or algorithmic parameters."""
        nonnegative = (
            "monomer_concentration_mol_l",
            "crosslinker_concentration_mol_l",
            "initiator_concentration_mol_l",
            "monomer_diffusion_nm2_s",
            "crosslinker_diffusion_nm2_s",
            "initiator_diffusion_nm2_s",
        )
        positive = (
            "box_length_nm",
            "time_step_s",
            "capture_radius_nm",
            "bond_length_nm",
        )
        probabilities = (
            "initiation_probability",
            "propagation_probability",
            "termination_combination_probability",
            "termination_disproportionation_probability",
        )
        for name in nonnegative + positive + probabilities:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if name in nonnegative and value < 0:
                raise ValueError(f"{name} must be nonnegative.")
            if name in positive and value <= 0:
                raise ValueError(f"{name} must be positive.")
            if name in probabilities and not 0 <= value <= 1:
                raise ValueError(f"{name} must lie between zero and one.")

        integers = ("crosslinker_functionality", "seed", "max_steps",
                    "snapshot_interval", "constraint_iterations")
        for name in integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if self.crosslinker_functionality < 2:
            raise ValueError(
                "Crosslinkers must have at least two reactive sites.")
        if self.seed < 0 or self.max_steps < 1:
            raise ValueError("seed >= 0 and max_steps >= 1 are required.")
        if self.snapshot_interval < 1 or self.constraint_iterations < 0:
            raise ValueError(
                "snapshot_interval >= 1; constraint_iterations >= 0.")
        if not isinstance(self.quench_at_end, bool):
            raise ValueError("quench_at_end must be Boolean.")
        if self.capture_radius_nm > 0.5 * self.box_length_nm:
            raise ValueError("Capture radius must not exceed half the box.")
        if self.bond_length_nm > self.capture_radius_nm:
            raise ValueError("Bond length must not exceed capture radius.")
        if (self.termination_combination_probability
                + self.termination_disproportionation_probability > 1):
            raise ValueError(
                "Termination probabilities must sum to <= 1.")

    def particle_counts(self) -> dict[str, int]:
        """Convert molar concentrations to particle counts in a cubic box."""
        volume_l = self.box_length_nm ** 3 * NM3_TO_LITERS
        scale = volume_l * AVOGADRO
        return {
            "monomer": int(math.floor(
                self.monomer_concentration_mol_l * scale + 0.5)),
            "crosslinker": int(math.floor(
                self.crosslinker_concentration_mol_l * scale + 0.5)),
            "initiator": int(math.floor(
                self.initiator_concentration_mol_l * scale + 0.5)),
        }

    @classmethod
    def from_json(cls, path: str | Path) -> SimulationConfig:
        """Construct and validate a configuration from a plain JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Configuration JSON must be an object.")
        return cls(**values)

    def to_dict(self) -> dict:
        """Return a JSON-compatible record of all simulation parameters."""
        return asdict(self)
