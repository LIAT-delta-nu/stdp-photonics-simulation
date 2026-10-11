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
