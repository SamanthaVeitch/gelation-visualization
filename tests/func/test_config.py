"""Input validation and concentration-to-count conversion tests."""

from dataclasses import replace
import json

import pytest

from gelation_sim import SimulationConfig
from gelation_sim.config import AVOGADRO, NM3_TO_LITERS


def test_count_conversion_uses_molarity_and_volume():
    config = SimulationConfig(
        monomer_concentration_mol_l=0.2,
        crosslinker_concentration_mol_l=0.02,
        initiator_concentration_mol_l=0.01,
        box_length_nm=10.0,
    )
    factor = AVOGADRO * NM3_TO_LITERS * 1000
    counts = config.particle_counts()
    assert counts["monomer"] == int(0.2 * factor + 0.5)
    assert counts["crosslinker"] == int(0.02 * factor + 0.5)
    assert counts["initiator"] == int(0.01 * factor + 0.5)


def test_invalid_parameters_are_rejected():
    base = SimulationConfig()
    invalid = (
        {"crosslinker_functionality": 1},
        {"seed": -2},
        {"max_steps": 0},
        {"snapshot_interval": 0},
        {"monomer_concentration_mol_l": -1.0},
        {"capture_radius_nm": 100.0},
        {"bond_length_nm": 1.7},
        {"initiation_probability": 1.3},
        {"termination_combination_probability": 0.9,
         "termination_disproportionation_probability": 0.9},
        {"quench_at_end": "false"},
        {"box_length_nm": float("nan")},
    )
    for changes in invalid:
        with pytest.raises(ValueError):
            replace(base, **changes)


def test_config_json_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    original = SimulationConfig(seed=777, max_steps=3)
    path.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    restored = SimulationConfig.from_json(path)
    assert original == restored
