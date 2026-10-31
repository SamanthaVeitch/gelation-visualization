"""Numerical analysis helpers for gelation simulation data."""
# Module imports
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

    if isinstance(end, bool) or not isinstance(end, int) or end <= start:
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
        kp = rp / conc_product if conc_product > 0 else None

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

    return numerator / denominator if denominator > 0 else None


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
        kt = rt / denom if denom > 0 else None

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
        if conv > max_conv or conc_squared <= 0:
            continue

        x = 2 * conc_squared
        numerator += x * rt
        denominator += x ** 2

    return numerator / denominator if denominator > 0 else None


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

    if any(a >= b for a, b in zip(steps, steps[1:])):
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

        fraction = remaining / init_num if init_num else None
        conc = calc_conc(remaining, config)
        results.append((step, remaining, fraction, conc))

    return results
