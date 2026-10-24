"""Pure data export for visualization and analysis collaborators.

This module intentionally contains no rendering, graph metrics, or plots.
Output CSV columns and file names are treated as a public integration API.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .model import Network
from .simulator import SimulationResult

PARTICLE_FIELDS = (
    "id", "species", "x_nm", "y_nm", "z_nm", "functionality",
    "active_radical", "bonded",
)
BOND_FIELDS = ("particle_a", "particle_b", "form_time")
EVENT_FIELDS = ("step", "reaction", "particle_a", "particle_b")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list) -> None:
    """Create a UTF-8 CSV with a stable header even for an empty dataset."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def export_network(network: Network, prefix: Path) -> tuple[Path, Path]:
    """Export all particle states and time-stamped bond records.

    The ``bonded`` column is a semicolon-separated adjacency list of IDs.
    ``active_radical`` is ``True`` or ``False``.  The separate bonds CSV
    contains one undirected edge per row with an integer formation step.
    """
    particle_path = prefix.parent / f"{prefix.name}_particles.csv"
    bond_path = prefix.parent / f"{prefix.name}_bonds.csv"
    particles = [
        (
            particle.id,
            particle.species,
            *[float(value) for value in particle.position],
            particle.functionality,
            particle.active_radical,
            ";".join(str(value) for value in sorted(particle.bonded)),
        )
        for particle in network.particles
    ]
    bonds = [
        (bond.particle_a, bond.particle_b, bond.form_time)
        for bond in network.bond_records
    ]
    _write_csv(particle_path, PARTICLE_FIELDS, particles)
    _write_csv(bond_path, BOND_FIELDS, bonds)
    return particle_path, bond_path


def export_result(
        result: SimulationResult,
        output_dir: str | Path) -> Path:
    """Write the complete run and snapshots into a self-contained folder.

    Existing outputs with matching names are overwritten.  Other files
    remain untouched.  A snapshot index lists the explicitly saved steps.
    """
    root = Path(output_dir)
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Avoid obsolete snapshots when overwriting an older run in this folder.
    # Remove only files bearing names reserved for generated snapshots.
    for pattern in ("step_*_particles.csv", "step_*_bonds.csv"):
        for stale_file in snapshots_dir.glob(pattern):
            stale_file.unlink()

    export_network(result.network, root / "network")
    _write_csv(root / "events.csv", EVENT_FIELDS, result.events_log)

    with (root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(result.config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    index = []
    for snapshot in result.snapshots:
        prefix = f"step_{snapshot.time:06d}"
        export_network(snapshot, snapshots_dir / prefix)
        index.append((
            snapshot.time,
            f"snapshots/{prefix}_particles.csv",
            f"snapshots/{prefix}_bonds.csv",
        ))
    _write_csv(
        root / "snapshot_index.csv",
        ("step", "particles_file", "bonds_file"),
        index,
    )

    # Specify enough metadata to interpret the files without simulator code.
    manifest = {
        "schema_version": "1.0",
        "seed": result.config.seed,
        "final_step": result.network.time,
        "box_length_nm": result.config.box_length_nm,
        "boundary_conditions": "periodic_xyz",
        "particle_ids": "zero_based_contiguous",
        "bond_edges": "undirected_unique",
        "time_unit": "simulation_step",
        "physical_time_per_step_s": result.config.time_step_s,
        "reaction_event_missing_partner": -1,
        "particle_count": len(result.network.particles),
        "bond_count": len(result.network.bond_records),
        "snapshot_count": len(result.snapshots),
        "event_count": len(result.events_log),
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return root


def load_network(
        particles_csv: str | Path,
        bonds_csv: str | Path,
        *,
        seed: int | None = None,
        time: int = 0) -> Network:
    """Rebuild a collaborator-ready Network from the public CSV schema.

    This is input/output integration, not a graphical or statistical module.
    Adjacency is regenerated from the bond table and checked against the
    redundant adjacency column to detect inconsistent/corrupted exports.
    """
    from .model import Particle

    with Path(particles_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["id"]))
    particles = []
    expected_adjacency = {}
    for row in rows:
        active = row["active_radical"]
        if active not in ("True", "False"):
            raise ValueError("Invalid Boolean in active_radical column.")
        particle_id = int(row["id"])
        particles.append(Particle(
            id=particle_id,
            species=row["species"],
            position=[
                float(row["x_nm"]),
                float(row["y_nm"]),
                float(row["z_nm"]),
            ],
            functionality=int(row["functionality"]),
            active_radical=(active == "True"),
        ))
        bonded = row["bonded"]
        expected_adjacency[particle_id] = (
            set(map(int, bonded.split(";"))) if bonded else set()
        )

    network = Network(particles, seed=seed, time=time)
    with Path(bonds_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            network.add_bond(
                int(row["particle_a"]),
                int(row["particle_b"]),
                int(row["form_time"]),
            )
    for particle in network.particles:
        if set(particle.bonded) != expected_adjacency[particle.id]:
            raise ValueError(
                f"Bond table disagrees with adjacency of {particle.id}.")
    return network


def load_result(output_dir: str | Path) -> SimulationResult:
    """Load a complete exported run including each Network snapshot."""
    from .config import SimulationConfig

    root = Path(output_dir)
    config = SimulationConfig.from_json(root / "config.json")
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["schema_version"] != "1.0":
        raise ValueError("Unsupported output schema version.")
    network = load_network(
        root / "network_particles.csv",
        root / "network_bonds.csv",
        seed=config.seed,
        time=int(manifest["final_step"]),
    )
    with (root / "events.csv").open(newline="", encoding="utf-8") as handle:
        events_log = [
            [
                int(row["step"]),
                row["reaction"],
                int(row["particle_a"]),
                int(row["particle_b"]),
            ]
            for row in csv.DictReader(handle)
        ]
    snapshots = []
    with (root / "snapshot_index.csv").open(
            newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            snapshots.append(load_network(
                root / row["particles_file"],
                root / row["bonds_file"],
                seed=config.seed,
                time=int(row["step"]),
            ))
    if not snapshots or snapshots[-1].time != network.time:
        raise ValueError("Snapshot index must end at the final network step.")
    return SimulationResult(
        network=network,
        snapshots=snapshots,
        events_log=events_log,
        config=config,
    )
