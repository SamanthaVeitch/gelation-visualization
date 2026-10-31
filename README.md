# Polymer Network Formation Simulator

A reproducible, coarse-grained **3D free-radical gelation simulator** for the
*Heterogeneous Gelation Visualization Software* group project. This repository
contains **structure generation and data export only**. 

## Model in brief

The simulator uses random Brownian displacement in a periodic cubic box,
local distance-limited reactions, and simple geometric bond-length correction.
It does **not** calculate an atomistic force field. Chemistry follows a
simplified vinyl radical polymerization mechanism:

1. **Initialize:** convert monomer, multifunctional crosslinker, and initiator
   concentrations (mol/L) into integer molecule counts within the box.
   Initialize reagent positions randomly using a seeded RNG.
2. **Diffuse:** apply independent Gaussian displacements with per-coordinate
   variance `2 * diffusion_coefficient * time_step`. Wrap at all boundaries.
   Relax existing covalent bonds toward an adjustable target length.
3. **Initiate:** each unused initiator receives one Bernoulli trial per step.
   Activated initiators can initiate exactly once.
4. **Terminate:** nearby pairs of active radicals can combine (creating a
   bond) or disproportionate (without creating a bond).
5. **Propagate:** active radicals can attack nearby unreacted vinyl groups.
   Form a bond, remove one available functional group from the *target*,
   remove the source radical, and activate a radical on the target.
6. **Record:** store each chemical reaction in an event log and deep-copy the
   network at configured snapshot steps.
7. **Finalize:** optionally quench radicals at the finite-time endpoint;
   save final network and all raw CSV files.

**Important assumptions:** each particle is one coarse-grained reagent, not
an atom; monomer has one initially available vinyl group; crosslinker has `f`
initially available vinyl groups; initiator has zero vinyl groups and can
activate once. `Particle.functionality` counts **remaining unreacted vinyl
groups**, not a particle's graph degree. Initiation and propagation transfer a
single radical, preserving exactly the roadmap's Boolean `active_radical`
field. For a multifunctional crosslinker, that constraint permits **at most
one simultaneous radical per entire node**. This suppresses concurrent chain
growth from different vinyl arms, and is a controlled simplification, not a
quantitatively complete kinetic treatment. A crosslinker with `f` vinyl groups
may acquire more than `f` neighbors because incorporated vinyl moieties can
have links on both sides of a polymer backbone. Solvent molecules, explicit
steric exclusion, chain transfer, realistic bond angles, entanglement physics,
and hydrodynamic interactions are not modeled. Bond projection maintains an
approximate geometry but does not guarantee physically admissible polymer
conformations. **Do not infer experimental gel points or absolute rates without
calibration and validation. The example parameters are selected to show
substantial connectivity in a modest-sized test box, not to represent a
specific chemical formulation.**

### Important probability definitions

All probabilities are discrete **per-step**, rather than kinetic rate
constants. `initiation_probability` is per unactivated initiator per step;
`propagation_probability` is per neighboring vinyl group per step. For a
crosslinker with `f` remaining groups, the encounter probability is
`1 - (1 - propagation_probability)**f`. The two termination probabilities
refer to mutually exclusive outcomes of one neighboring radical-pair trial,
and their sum must be at most 1. A particle makes at most one propagation or
termination reaction per step. A radical newly created by propagation is not
eligible to propagate until the next step. The termination trial occurs before
propagation. A partner ID of `-1` denotes a single-particle event.

`quench_at_end=true` is an *administrative boundary condition* at the end of
a finite run, not a calibrated physical termination mechanism. It causes
`termination_quench` events and clears remaining radicals in the final output.
Set it to `false` if an unfinished network with active radicals is desired.

## Installation

Recommended: Python 3.11, in a clean Anaconda or virtual environment.
From the repository root (works in Anaconda Prompt on Windows):

```bash
python -m pip install -e ".[dev]"
```

Or use only runtime packages with `python -m pip install -e .`.

For Anaconda, an alternative is `conda env create -f environment.yml`,
`conda activate gelation-sim`, followed by `python -m pip install -e .`.

## Run a simulation

```bash
python -m gelation_sim --config config/example.json --output runs/run_001
```

Or, with built-in default parameters:

```bash
gelation-sim --output runs/quickstart
```

Edit a copy of `config/example.json` to adjust chemistry, geometry, diffusion,
number of steps, snapshot cadence, probabilities, or reproducibility seed.
Input counts are `floor(C_mol/L * L_nm**3 * 1e-24 L/nm^3 * N_A + 0.5)`,
where `N_A = 6.02214076e23 mol^-1`. For example, in a 12 nm cube at 0.4 M,
approximately 416 monomer nodes are simulated (not Avogadro-scale counts).
A small concentration may round to zero in a small simulation box; increase
the box size or concentration as appropriate. `seed` determines stochastic
repeatability with the same dependency versions and operating environment.

### Python API

```python
from gelation_sim import (
    SimulationConfig, Simulator, export_result, load_result,
)

config = SimulationConfig.from_json("config/example.json")
result = Simulator(config).run()
export_result(result, "runs/run_001")

final_network = result.network
snapshots = result.snapshots  # independent Network copies
chemical_events = result.events_log

# On another computer, load exactly these same public data objects:
reloaded = load_result("runs/run_001")

# Collaborators can use the public Network interface directly:
particles = final_network.particles  # np.ndarray[Particle], shape (N,)
edges = final_network.bonds        # np.ndarray[int], shape (M, 2)
step = final_network.time
```

## Public interfacial contract

| Public attribute | Type | Exact interpretation |
| --- | --- | --- |
| `Network.particles` | NumPy 1-D `object` array | Complete list of `Particle`, indexed by ID |
| `Network.bonds` | NumPy integer `(M, 2)` array | Unique undirected `(particle_a, particle_b)` edges; read-only |
| `Network.time` | int | Discrete step represented by this network |
| `Network.seed` | int | Seed used when created |
| `Particle.id` | int | Zero-based, contiguous molecule ID |
| `Particle.species` | str | `monomer`, `crosslinker`, or `initiator` |
| `Particle.position` | NumPy float array, `(3,)` | `[x, y, z]` in nanometers |
| `Particle.functionality` | int | Number of remaining reactive vinyl groups |
| `Particle.active_radical` | bool | Whether one radical is present on this coarse-grained node |
| `Particle.bonded` | `list[int]` | IDs of all adjacent particles |
| `Bond.particle_a` | int | First endpoint ID |
| `Bond.particle_b` | int | Second endpoint ID |
| `Bond.form_time` | int | Bond formation step, not elapsed seconds |
| `Network.bond_records` | `list[Bond]` | Additional metadata without changing `Network.bonds` |
| `SimulationResult.snapshots` | `list[Network]` | Independent full network copies at selected steps |
| `SimulationResult.events_log` | list of 4-item lists | `[step, reaction, particle_a, particle_b]` |

Supported event labels: `initiation`, `propagation`,
`termination_combination`, `termination_disproportionation`, and
`termination_quench`. The event log captures **chemical events**, not every
Brownian displacement. Do not append directly to `Network.bonds` or edit
`Particle.bonded`; use `Network.add_bond(...)` to preserve consistency. The
network records each undirected bond once, but both particles list each other.

## Numerical kinetic analysis

`gelation\_sim.analysis` provides read-only helpers for analyzing `Network`
snapshots, the chronological reaction-event log, and their associated
`SimulationConfig`. Analysis is separate from the simulator and is output into two CSV files. Functions can be used with either a
fresh `SimulationResult` or one reconstructed by `load\_result()`.

### Available calculations

|Function|Calculation or result|
|-|-|
|`calc\_conc(num, config)`|Particle count converted to concentration (mol/L) using the simulation box volume.|
|`calc\_time(step, config)`|Simulation step converted to elapsed seconds.|
|`calc\_interval(start, end, config)`|Time elapsed between two simulation steps, in seconds.|
|`calc\_init(steps, events, config)`|Remaining initiator count, fraction, and concentration at requested steps.|
|`calc\_kd(init\_data, config)`|Effective initiator activation coefficient (s^-1) from the slope of log remaining-initiator fraction versus time.|
|`calc\_half\_life(kd)`|Fitted initiator half-life, `ln(2) / kd`, in seconds.|
|`calc\_init\_rate(steps, events, config)`|Observed initiation-event rate in mol/(L s) over consecutive step intervals.|
|`calc\_init\_eff(events)`|Fraction of *resolved* initiations that successfully propagate.|
|`calc\_rad(snapshots, events, config)`|Active radical count and concentration at every snapshot.|
|`calc\_kp(snapshots, events, config)`|Interval propagation counts, observed polymerization rates, and effective propagation coefficients.|
|`fit\_kp(kp\_data)`|Single effective propagation coefficient from a least-squares fit through the origin.|
|`calc\_kt(snapshots, events, config)`|Interval conventional termination counts, radical disappearance rates, effective termination coefficients, and vinyl conversion.|
|`fit\_kt(kt\_data, max\_conv=0.10)`|Low-conversion effective termination coefficient from a least-squares fit through the origin.|
|`calc\_rad\_ss(conc, kd, kt, eff)`|**Theoretical** steady-state radical concentration (mol/L).|
|`calc\_poly\_rate\_ss(kp, mon\_conc, rad\_ss)`|**Theoretical** steady-state polymerization rate (mol/(L s)).|
|`calc\_chain\_len(steps, events)`|Aggregate kinetic chain-length estimate over consecutive step intervals.|
|`calc\_chain\_len\_run(events)`|Aggregate kinetic chain-length estimate over the complete event log.|

The `steps` arguments use discrete simulation steps, not seconds. For interval
calculations, the starting step is excluded and the ending step is included.
Event logs must be chronological; snapshots must be ordered by simulation step.
A rate for an interval requires at least two steps or snapshots.


The functions return numerical values or lists of tuples; they do not modify
snapshots or write CSV files. A fitted coefficient or theoretical prediction
may be `None` when the required observations are insufficient or the
calculation is undefined. `calc\_half\_life(None)` returns `None`, while zero or
negative `kd` raises `ValueError` in the current implementation.

### Scientific interpretation and limitations

* **Observed versus predicted:** initiator depletion, initiation rates,
radical concentrations, propagation rates, and termination rates are
calculated from simulation records. `calc\_rad\_ss()` and
`calc\_poly\_rate\_ss()` are **theoretical steady-state predictions** used for
comparison, not additional measurements of the same quantities.
* **Effective rate coefficients:** `kd`, `kp`, and `kt` are estimated from the
simulated trajectory, not supplied experimental kinetic constants. The
simulator uses per-step probabilities and spatial encounter rules; fitted
coefficients depend on these modeling choices. Interval `kp` and `kt`
calculations approximate concentrations using snapshot endpoints.
* **Low-conversion termination fit:** `fit\_kt()` defaults to `max\_conv=0.10`,
retaining intervals ending at or below 10% overall vinyl-group conversion.
This cutoff is an analysis choice, not a universal physical threshold.
Network formation and spatial restrictions may change effective `kt` later.
* **Initiator efficiency:** `calc\_init\_eff()` returns successful initiations
divided by successful plus unsuccessful *resolved* initiations. Quenched
and still-active attempts are excluded. This is not conventional chemical
initiator efficiency: the simulator creates one radical per activation,
not two radical fragments per decomposed initiator. The theoretical
steady-state prediction uses this model-specific fraction as an effective
approximation.
* **Final-step quenching:** `calc\_rad()` and `calc\_kt()` correct the final
radical count for administrative quench events where applicable, so their
final radical concentrations represent the state before quenching.
Administrative quenching is excluded from conventional termination rates
and kinetic chain-length denominators.
* **Kinetic chain length:** `calc\_chain\_len()` and `calc\_chain\_len\_run()` use
propagation counts divided by twice the number of conventional termination
events. These are aggregate estimates, not measured lengths of individual
radical lineages. Propagations from unfinished lineages may be included.
Results are `None` when no conventional terminations occur.


## Output structure

```text
gelation_project/
├── config/
│   └── example.json
├── src/gelation_sim/
│   ├── __init__.py
│   ├── __main__.py
│   ├── analysis.py
│   ├── config.py
│   ├── model.py
│   ├── simulator.py
│   └── export.py
├── tests/
├── .github/workflows/
│   ├── func_tests.yml
│   ├── style_check.yml
│   └── unit_tests.yml
├── pyproject.toml
├── README.md
└── runs/                         # created on demand; ignored by Git
    └── run_001/
        ├── config.json
        ├── manifest.json
        ├── network_particles.csv
        ├── network_bonds.csv
        ├── events.csv
        ├── snapshot_index.csv
        └── snapshots/
            ├── step_000000_particles.csv
            ├── step_000000_bonds.csv
            ├── step_000020_particles.csv
            ├── step_000020_bonds.csv
            └── ...
```

### CSV schema for collaborating students

- `network_particles.csv`: `id,species,x_nm,y_nm,z_nm,functionality,
  active_radical,bonded`. `bonded` is a **semicolon-delimited** ID string
  inside a single cell. All coordinate values are in nanometers.
- `network_bonds.csv`: `particle_a,particle_b,form_time` (undirected
  edges, each listed exactly once; endpoints in ascending ID order).
- `events.csv`: `step,reaction,particle_a,particle_b`. The partner is
  `-1` for initiation and terminal quench events; no bond is created
  for disproportionation or quench.
- `snapshot_index.csv`: `step,particles_file,bonds_file` with paths relative
  to the run output directory. Each snapshot has the same particle and bond
  schema as the final output.
- `manifest.json`: schema version, units, periodic geometry, IDs, seed, final
  step, and output sizes. `config.json` preserves all model assumptions.

- `validation_constants.csv`: `Name,value,units`. Contains the non-time dependent calculated values and the values given during configuration.
- `validation_equations.csv`: `Time,[I],[M],[P radical],Ni,Nm,Nr,InitiatorFractionRemaining,InitiationRate,RadicalDisappearanceRate,SimulatedPolymerizationRate` The rows contain the values at each snap shot.

The public `load_result(path)` function reconstructs `Network`, `Bond`,
`Particle`, `snapshots`, and `events_log` directly from these files. For just
one CSV pair, call `load_network(particles_csv, bonds_csv, seed=...,
time=...)`. This lets other team members stay independent of the reaction
simulation implementation. Overwriting a run removes obsolete files bearing
reserved generated snapshot filenames; unrelated files are untouched.

Snapshots always contain step zero and the final step, even if the latter
falls between regular capture intervals. At the end of a run, the final
snapshot matches the finalized `Network` including quenching if requested.

**Periodic boundary note:** positions in all CSV files lie in `[0, L)`; two
bonded particles can appear to lie on opposite faces of the box. Rendering
and spatial-distance calculations must use minimum-image periodic distances.

## Tests and style

Run at repository root:

```bash
python -m pytest -q
python -m pycodestyle src tests
```

GitHub Actions runs both commands on each push and pull request, and when
manually dispatched. All project Python code follows **PEP 8** (the Python
style convention), using module and function docstrings, explanatory inline
comments, and type annotations. Tests cover the data contract, counts and
validation, radical mechanisms, periodic encounters, seeded repeatability,
independent snapshots, and output CSV schemas.

## Model scope and next scientific steps

The current method is a transparent, mesoscopic *prototype*. In particular,
reactant counts may be large relative to a molecular-scale box, geometry is
not sterically self-avoiding, and any radical reactivity and diffusion
parameters should be calibrated or systematically varied before using the
results as physical predictions. Possible future simulation-only upgrades
include explicit steric constraints, multiple radical sites per crosslinker
(which **would require an agreed interface change**), and event-rate-based
kinetics. Visualization and gel/chain statistics belong to their separately
maintained modules; they should read the exported files or the shared
`Network` class, without modifying simulator state.
