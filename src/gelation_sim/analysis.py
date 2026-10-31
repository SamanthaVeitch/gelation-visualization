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
