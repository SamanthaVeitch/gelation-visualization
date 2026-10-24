"""Command-line entry point for gelation_sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SimulationConfig
from .export import export_result
from .simulator import Simulator


def main() -> None:
    """Parse arguments, run the simulation, and export its raw structure."""
    parser = argparse.ArgumentParser(
        description="Simulate coarse-grained radical polymer gelation.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a JSON configuration (defaults to built-in values).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/example"),
        help="Directory receiving network, events, and snapshots CSV files.",
    )
    args = parser.parse_args()
    config = (
        SimulationConfig.from_json(args.config)
        if args.config is not None else SimulationConfig()
    )
    result = Simulator(config).run()
    directory = export_result(result, args.output)
    print(f"Output: {directory.resolve()}")
    print(f"Particles: {len(result.network.particles)}")
    print(f"Bonds: {len(result.network.bond_records)}")
    print(f"Snapshots: {len(result.snapshots)}")
    print(f"Reaction events: {len(result.events_log)}")


if __name__ == "__main__":
    main()
