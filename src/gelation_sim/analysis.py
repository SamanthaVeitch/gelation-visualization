"""Numerical analysis helpers for gelation simulation data."""
# Module imports
import math
import numpy as np

# Script imports
from .config import AVOGADRO, NM3_TO_LITERS


# Basic helpers
def calc_conc(num, config):
    """Calculate particle concentration in mol/L using the box volume."""
    if isinstance(num, bool) or not isinstance(num, int) or num < 0:
        raise ValueError("num must be a nonnegative integer.")

    vol = NM3_TO_LITERS * config.box_length_nm ** 3
    return num / (AVOGADRO * vol)


def calc_time(step, config):
    """Calculate elapsed simulation time in seconds."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer.")

    return step * config.time_step_s


def calc_interval(start, end, config):
    """Calculate elapsed seconds between simulation steps."""
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a nonnegative integer.")

    elif isinstance(end, bool) or not isinstance(end, int) or end <= start:
        raise ValueError("end must be an integer greater than start.")

    return calc_time(end - start, config)


# Calculate constants
def calc_kd(init_data, config):
    """Estimate kd from the slope of log initiator fraction versus time.

    Parameters:
        init_data: List of tuples returned by calc_init().
            needs large initial initiator population

    Returns:
        Effective kd in s^-1, or None if insufficient data exists.
    """
    time = []
    log_frac = []

    for step, _, fraction, _ in init_data:
        if fraction is not None and fraction > 0:
            time.append(calc_time(step, config))
            log_frac.append(np.log(fraction))

    if len(time) < 2:
        return None

    slope, intercept = np.polyfit(time, log_frac, 1)

    return -float(slope)


def calc_kp(snapshots, events, config):
    """Estimate effective kp over consecutive snapshot intervals.

    Returns:
        List of tuples (start, end, prop_rxn_num, rp, kp).
    """
    if len(snapshots) < 2:
        return []

    results = []
    event_idx = 0

    for prev, curr in zip(snapshots, snapshots[1:]):
        dt = calc_interval(prev.time, curr.time, config)
        prop_rxn_num = 0

        while event_idx < len(events) and events[event_idx][0] <= curr.time:
            if events[event_idx][1] == "propagation":
                prop_rxn_num += 1
            event_idx += 1

        mon_prev = sum(p.functionality for p in prev.particles)
        mon_curr = sum(p.functionality for p in curr.particles)
        rad_prev = sum(p.active_radical for p in prev.particles)
        rad_curr = sum(p.active_radical for p in curr.particles)

        mon_conc = calc_conc(mon_prev + mon_curr, config) / 2
        rad_conc = calc_conc(rad_prev + rad_curr, config) / 2

        rp = calc_conc(prop_rxn_num, config) / dt
        denom = mon_conc * rad_conc
        conc_product = mon_conc * rad_conc

        if conc_product <= 0:
            kp = None
        else:
            kp = rp / conc_product

        results.append((prev.time,
                        curr.time,
                        prop_rxn_num,
                        rp,
                        conc_product,
                        kp))

    return results


def fit_kp(kp_data):
    """Fit effective kp from interval propagation rates."""
    numerator = 0.0
    denominator = 0.0

    for start, end, prop_rxn_num, rp, conc_product, kp in kp_data:
        if conc_product <= 0:
            continue

        numerator += conc_product * rp
        denominator += conc_product ** 2

    if denominator <= 0:
        return None
    else:
        return numerator / denominator


def calc_kt(snapshots, events, config):
    """Calculate effective kt over consecutive snapshot intervals.

    Administrative quenching is excluded from termination rates.
    """
    if len(snapshots) < 2:
        return []

    init_vinyl_num = sum(
        p.functionality for p in snapshots[0].particles
    )
    results = []
    event_idx = 0

    for prev, curr in zip(snapshots, snapshots[1:]):
        dt = calc_interval(prev.time, curr.time, config)
        term_rxn_num = 0
        quench_num = 0

        while event_idx < len(events) and events[event_idx][0] <= curr.time:
            reaction = events[event_idx][1]

            if reaction in (
                "termination_combination",
                "termination_disproportionation",
            ):
                term_rxn_num += 1
            elif reaction == "termination_quench":
                quench_num += 1

            event_idx += 1

        rad_prev = sum(p.active_radical for p in prev.particles)
        rad_curr = sum(p.active_radical for p in curr.particles)

        if curr.time == config.max_steps and config.quench_at_end:
            rad_curr += quench_num

        conc_prev = calc_conc(rad_prev, config)
        conc_curr = calc_conc(rad_curr, config)

        conc_squared = (conc_prev ** 2 + conc_curr ** 2) / 2
        rt = 2 * calc_conc(term_rxn_num, config) / dt

        denom = 2 * conc_squared
        if denom <= 0:
            kt = None
        else:
            kt = rt / denom

        vinyl_num = sum(p.functionality for p in curr.particles)
        conv = (
            1 - vinyl_num / init_vinyl_num
            if init_vinyl_num > 0 else None
        )

        results.append((
            prev.time, curr.time, term_rxn_num,
            rt, conc_squared, kt, conv,
        ))

    return results


def fit_kt(kt_data, max_conv=0.10):
    """Fit effective kt using low-conversion termination data.

    A constant effective kt is more appropriate at low conversion.
    Spatial crowding and network formation may change kt later.
    The conversion cutoff is for ease of analysis.
    """
    if not 0 <= max_conv <= 1:
        raise ValueError("max_conv must lie between zero and one.")

    numerator = 0.0
    denominator = 0.0

    for start, end, term_rxn_num, rt, conc_squared, kt, conv in kt_data:
        if conv is None or conv > max_conv or conc_squared <= 0:
            continue

        x = 2 * conc_squared
        numerator += x * rt
        denominator += x ** 2

    if denominator <= 0:
        return None
    else:
        return numerator / denominator


def calc_init_eff(events):
    """Calculate the fraction of resolved initiations that propagate.

    Returns:
        Fraction of successful initiations
        over (successful initiations + successful initiations),
        or None if no initiations have resolved.
    """
    active = set()
    successful = 0
    unsuccessful = 0

    for step, reaction, a, b in events:
        if reaction == "initiation":
            active.add(a)

        elif reaction == "propagation" and a in active:
            active.remove(a)
            successful += 1

        elif reaction in (
            "termination_combination",
            "termination_disproportionation",
        ):
            for particle_id in (a, b):
                if particle_id in active:
                    active.remove(particle_id)
                    unsuccessful += 1

        elif reaction == "termination_quench":
            active.discard(a)

    resolved = successful + unsuccessful

    if resolved == 0:
        return None
    else:
        return successful / resolved


# Initiator calculations
def calc_init(steps, events, config):
    """Calculate initiator depletion at specified simulation steps.

    Parameters:
        steps: List of nonnegative integer simulation steps, sorted in
            ascending order. May contain one or multiple steps.

    Returns:
        List of tuples (step, remaining, fraction, concentration).
    """
    if any(not isinstance(step, int) or step < 0 for step in steps):
        raise ValueError("steps must contain nonnegative integers.")

    elif any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("steps must be in strictly ascending order.")

    init_num = config.particle_counts()["initiator"]
    remaining = init_num
    event_idx = 0
    results = []

    for step in steps:
        while event_idx < len(events) and events[event_idx][0] <= step:
            if events[event_idx][1] == "initiation":
                remaining -= 1
            event_idx += 1

        if remaining < 0:
            raise ValueError("Initiation events exceed initial initiators.")

        if init_num == 0:
            fraction = None
        else:
            fraction = remaining / init_num
        conc = calc_conc(remaining, config)
        results.append((step, remaining, fraction, conc))

    return results


def calc_half_life(kd):
    """Calculate initiator half-life in seconds."""
    if kd is None:
        return None
    elif kd <= 0:
        raise ValueError("kd must be positive.")
    else:
        return math.log(2) / kd


"""
def calc_init_rate_theor(conc, kd, eff):
    #Calculate theoretical initiation rate in mol/(L s).
    if conc < 0 or kd < 0 or not 0 <= eff <= 1:
        raise ValueError("Invalid concentration, kd, or efficiency.")

    return 2 * eff * kd * conc
"""


def calc_init_rate(steps, events, config):
    """Calculate simulated initiation rates between consecutive steps."""
    if any(not isinstance(step, int) or step < 0 for step in steps):
        raise ValueError("steps must contain nonnegative integers.")
    elif any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("steps must be in strictly ascending order.")

    results = []
    event_idx = 0

    for start, end in zip(steps, steps[1:]):
        while event_idx < len(events) and events[event_idx][0] <= start:
            event_idx += 1

        init_num = 0

        while event_idx < len(events) and events[event_idx][0] <= end:
            if events[event_idx][1] == "initiation":
                init_num += 1

            event_idx += 1

        dt = calc_interval(start, end, config)
        rate = calc_conc(init_num, config) / dt

        results.append((start, end, init_num, rate))

    return results


# Radical calculations
def calc_rad(snapshots, events, config):
    """Calculate active radical counts and concentrations at snapshots."""
    quench_num = 0

    if config.quench_at_end:
        quench_num = sum(
            1 for step, reaction, a, b in events
            if step == config.max_steps
            and reaction == "termination_quench"
        )

    results = []

    for snapshot in snapshots:
        rad_num = sum(
            p.active_radical for p in snapshot.particles
        )

        if config.quench_at_end and snapshot.time == config.max_steps:
            rad_num += quench_num

        conc = calc_conc(rad_num, config)
        results.append((snapshot.time, rad_num, conc))

    return results


def calc_rad_ss(conc, kd, kt, eff):
    """Calculate theoretical steady-state radical concentration."""
    if any(value is None for value in (conc, kd, kt, eff)):
        return None
    elif kt == 0:
        return None
    elif conc < 0 or kd < 0 or kt < 0 or not 0 <= eff <= 1:
        raise ValueError("Invalid steady-state calculation inputs.")

    return math.sqrt(eff * kd * conc / kt)


# Polymer calculations
def calc_poly_rate_ss(kp, mon_conc, rad_ss):
    """Calculate theoretical steady-state polymerization rate."""
    if any(value is None for value in (kp, mon_conc, rad_ss)):
        return None
    elif kp < 0 or mon_conc < 0 or rad_ss < 0:
        raise ValueError("Inputs must be nonnegative.")

    return kp * mon_conc * rad_ss


def calc_chain_len(steps, events):
    """Calculate aggregate kinetic chain length by simulation interval.

    Uses events within steps, not individual radical lineages.
    Quenching is excluded.
    """
    if any(not isinstance(step, int) or step < 0 for step in steps):
        raise ValueError("steps must contain nonnegative integers.")

    elif any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("steps must be in strictly ascending order.")

    results = []
    event_idx = 0

    for start, end in zip(steps, steps[1:]):
        while event_idx < len(events) and events[event_idx][0] <= start:
            event_idx += 1

        prop_num = 0
        term_num = 0

        while event_idx < len(events) and events[event_idx][0] <= end:
            reaction = events[event_idx][1]

            if reaction == "propagation":
                prop_num += 1
            elif reaction in (
                "termination_combination",
                "termination_disproportionation",
            ):
                term_num += 1

            event_idx += 1

        chain_len = prop_num / (2 * term_num) if term_num else None
        results.append((start, end, prop_num, term_num, chain_len))

    return results


def calc_chain_len_run(events):
    """Calculate completed-run aggregate kinetic chain length.

    Administrative quenching is excluded.
    """
    prop_num = 0
    term_num = 0

    for step, reaction, a, b in events:
        if reaction == "propagation":
            prop_num += 1
        elif reaction in (
            "termination_combination",
            "termination_disproportionation",
        ):
            term_num += 1

    return prop_num / (2 * term_num) if term_num else None


# Data export
def calc_kinetic_params(result):
    """Calculate and retain shared run-level kinetic results."""
    snapshots = result.snapshots
    events = result.events_log
    config = result.config

    steps = [snapshot.time for snapshot in snapshots]
    init_data = calc_init(steps, events, config)
    rad_data = calc_rad(snapshots, events, config)
    init_rates = calc_init_rate(steps, events, config)
    kp_data = calc_kp(snapshots, events, config)
    kt_data = calc_kt(snapshots, events, config)

    kd = calc_kd(init_data, config)
    kp = fit_kp(kp_data)
    kt = fit_kt(kt_data) if kt_data else None
    eff = calc_init_eff(events)

    half_life = (
        calc_half_life(kd)
        if kd is not None and kd > 0 else None
    )

    return {
        "init_data": init_data,
        "rad_data": rad_data,
        "init_rates": init_rates,
        "kp_data": kp_data,
        "kt_data": kt_data,
        "kd": kd,
        "kp": kp,
        "kt": kt,
        "init_eff": eff,
        "half_life": half_life,
        "chain_len_run": calc_chain_len_run(events),
    }


def calc_validation_rows(result, params):
    """Assemble snapshot values for validation_equations.csv."""
    snapshots = result.snapshots
    config = result.config

    if not snapshots:
        return []

    init_data = params["init_data"]
    rad_data = params["rad_data"]
    init_rates = params["init_rates"]
    kp_data = params["kp_data"]
    kt_data = params["kt_data"]

    kd = params["kd"]
    kp = params["kp"]
    kt = params["kt"]
    eff = params["init_eff"]

    rows = []

    for i, snapshot in enumerate(snapshots):
        step, ni, fraction, init_conc = init_data[i]
        _, nr, rad_conc = rad_data[i]

        nm = sum(p.functionality for p in snapshot.particles)
        mon_conc = calc_conc(nm, config)

        init_rate = None
        term_rate = None
        poly_rate = None

        if i > 0:
            init_rate = init_rates[i - 1][3]
            term_rate = kt_data[i - 1][3]
            poly_rate = kp_data[i - 1][3]

        rad_ss = calc_rad_ss(init_conc, kd, kt, eff)
        poly_ss = calc_poly_rate_ss(kp, mon_conc, rad_ss)

        rows.append((
            f"step_{step:06d}",
            calc_time(step, config),
            init_conc,
            mon_conc,
            rad_conc,
            ni,
            nm,
            nr,
            fraction,
            init_rate,
            term_rate,
            poly_rate,
            rad_ss,
            poly_ss,
        ))

    return rows


def calc_validation_constants(result, params):
    """Assemble run-level constants in the validation workbook order.

    Returns:
        List of (name, value, units) tuples.
        Deferred quantities are None. The steady-state radical
        prediction uses the initial initiator concentration.
    """
    config = result.config
    volume = NM3_TO_LITERS * config.box_length_nm ** 3

    init_data = params["init_data"]
    init_conc = init_data[0][3] if init_data else None

    rad_ss = calc_rad_ss(
        init_conc,
        params["kd"],
        params["kt"],
        params["init_eff"],
    )

    return [
        ("Seed", config.seed, None),
        ("Snapshot interval", config.snapshot_interval, "steps"),
        ("Maximum steps", config.max_steps, "steps"),
        ("Time per step", config.time_step_s, "s"),
        ("Total time", calc_time(result.network.time, config), "s"),
        ("Constraint iterations", config.constraint_iterations, "count"),
        ("Capture radius", config.capture_radius_nm, "nm"),
        ("Bond length", config.bond_length_nm, "nm"),
        ("Box length", config.box_length_nm, "nm"),
        ("Volume", volume, "L"),
        ("Crosslinker functionality",
         config.crosslinker_functionality, "groups/particle"),
        ("Monomer diffusion",
         config.monomer_diffusion_nm2_s, "nm^2/s"),
        ("Crosslinker diffusion",
         config.crosslinker_diffusion_nm2_s, "nm^2/s"),
        ("Initiator diffusion",
         config.initiator_diffusion_nm2_s, "nm^2/s"),
        ("Initiation probability",
         config.initiation_probability, "per step"),
        ("Propagation probability",
         config.propagation_probability, "per encounter per step"),
        ("Termination combination probability",
         config.termination_combination_probability, "per pair trial"),
        ("Termination disproportionation probability",
         config.termination_disproportionation_probability,
         "per pair trial"),
        ("Quench at end", config.quench_at_end, "boolean"),
        ("kd", params["kd"], "1/s"),
        ("kp", params["kp"], "L/(mol*s)"),
        ("kt", params["kt"], "L/(mol*s)"),
        ("Initiator efficiency", params["init_eff"], "dimensionless"),
        ("Initiator half-life", params["half_life"], "s"),
        ("Predicted steady-state radical concentration",
         rad_ss, "mol/L"),
        ("Completed-run kinetic chain length",
         params["chain_len_run"], "propagations/chain"),
        ("Predicted mean kinetic chain length",
         None, "propagations/chain"),
        ("Mn", None, "g/mol"),
        ("Mw", None, "g/mol"),
        ("PDI", None, "dimensionless"),
    ]
