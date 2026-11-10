import numpy as np

# ============================================================
# Inputs
# ============================================================
def gaussian_pulse(
    t,
    power_peak,
    pulse_width,   # FWHM
    t_peak,
):
    """Gaussian pulse."""
    return power_peak * np.exp(
        -4.0 * np.log(2.0) *
        ((t - t_peak) / pulse_width) ** 2
    )

def gaussian_pulse_energy(
    power_peak,     #
    pulse_width,    # FWHM
):
    return power_peak * pulse_width * np.sqrt(np.pi /np.log(2)) / 2


# ============================================================
# Measurements
# ============================================================
import numpy as np


def pulse_energy(P: np.ndarray, t: np.ndarray) -> float:
    """Calculate pulse energy [J] from power [W] and time [s]."""
    return np.trapezoid(P, t)


def measure_pulse(P: np.ndarray, t: np.ndarray) -> dict:
    """
    Measure pulse peak, energy, and FWHM.

    Parameters
    ----------
    P : np.ndarray
        Pulse power [W].
    t : np.ndarray
        Time samples [s], possibly non-uniformly spaced.

    Returns
    -------
    dict
        t_peak      : Peak time [s]
        P_peak      : Peak power [W]
        t_HM_left   : Left half-maximum crossing [s]
        t_HM_right  : Right half-maximum crossing [s]
        fwhm        : Full width at half maximum [s]
        energy      : Energy of complete signal [J]
    """
    
    energy = pulse_energy(P, t)

    peak_idx = np.argmax(P)
    P_peak = P[peak_idx]
    t_peak = t[peak_idx]

    half_max = P_peak / 2
    left_idx = np.where(P[:peak_idx] <= half_max)[0]
    right_idx = np.where(P[peak_idx + 1:] <= half_max)[0]

    if len(left_idx) == 0 or len(right_idx) == 0:
        return {
            "t_peak": t_peak,
            "P_peak": P_peak,
            "t_HM_left": np.nan,
            "t_HM_right": np.nan,
            "fwhm": np.nan,
            "energy":energy,
        }
    else:
        i_left = left_idx[-1]
        i_right = peak_idx + 1 + right_idx[0]

        def crossing_time(i):
            return t[i] + (
                (half_max - P[i]) / (P[i + 1] - P[i])
            ) * (t[i + 1] - t[i])

        t_HM_left = crossing_time(i_left)
        t_HM_right = crossing_time(i_right - 1)

        fwhm = t_HM_right - t_HM_left

    return {
        "t_peak": t_peak,
        "P_peak": P_peak,
        "t_HM_left": t_HM_left,
        "t_HM_right": t_HM_right,
        "fwhm": fwhm,
        "energy":energy,
    }