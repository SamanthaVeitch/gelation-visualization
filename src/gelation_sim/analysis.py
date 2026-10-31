"""Numerical analysis helpers for gelation simulation data."""

from .config import AVOGADRO, NM3_TO_LITERS


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
