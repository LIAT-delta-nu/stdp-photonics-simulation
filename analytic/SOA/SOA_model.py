import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# Physical constants
# ============================================================

q = 1.602176634e-19       # C
h = 6.62607015e-34        # J s
c = 299792458.0           # m/s


# ============================================================
# SOA parameters from Li et al. (2008)
# ============================================================

A = 2.0e8                 # 1/s
B = 3.83e-17              # m^3/s
C = 4.0e-41               # m^6/s



n0 = 1.0e24               # m^-3

alpha = 1.4e4             # m^-1

beta_eff = 7.3e-3

# Choose one of the devices in the paper
L = 800e-6                # 800 um
I = 72e-3                 # 72 mA

W = 2.0e-6                # active-region width [m]
d = 0.2e-6                # active-region thickness [m]


# ============================================================
# Simulation parameters
# ============================================================

Nz = 100

z = np.linspace(0.0, L, Nz)
dz = z[1] - z[0]

# Optical wavelengths
lambda_pump = 1551e-9
lambda_probe = 1550e-9


# ============================================================
# Pump pulse
# ============================================================

pump_power_peak = 5e-3       # W
pump_pulse_width = 65e-12    # s
pump_t0 = 200e-12             # s


def pump_input_power(t):
    """Gaussian pump pulse."""
    return pump_power_peak * np.exp(
        -4.0 * np.log(2.0) *
        ((t - pump_t0) / pump_pulse_width) ** 2
    )


# Weak CW probe
probe_input_power = 40e-6     # W


# ============================================================
# Gain model from Li et al.
# ============================================================

def modal_gain(
    n, 
    wavelength, 
    lambda_0 = 1.585e-6, # m
    a0 = 4.5e-21,       # m^2
    a1 = 1.5e18,        # m^-3
    a2 = 5.0e-33,       # m^4
):
    """
    Modal gain g_m in 1/m.

    g_m = a0 (n - n0)
          - a1 (lambda - lambda_n)^2

    lambda_n = lambda0 - a2 (n - n0)
    """

    

    lambda_n = lambda_0 - a2 * (n - n0)

    return (
        a0 * (n - n0)
        - a1 * (wavelength - lambda_n) ** 2
    )


# ============================================================
# Optical propagation
# ============================================================

def propagate_power(P_in, n, wavelength):
    """
    Propagate a forward travelling optical signal
    through the SOA for the current carrier-density profile.
    """

    P = np.empty(Nz)
    P[0] = P_in

    for i in range(Nz - 1):

        g = modal_gain(n[i], wavelength)

        # dP/dz = (g - alpha) P
        P[i + 1] = P[i] * np.exp(
            (g - alpha) * dz
        )

    return P


# ============================================================
# Carrier-density dynamics
# ============================================================

def dn_dt(t, n):
    """
    dn/dt for every longitudinal SOA section.
    """

    # Optical powers at this instant
    P_pump = propagate_power(
        pump_input_power(t),
        n,
        lambda_pump
    )

    P_probe = propagate_power(
        probe_input_power,
        n,
        lambda_probe
    )

    # Total stimulated-emission contribution
    #
    # For the present simplified model we use the
    # pump and probe powers directly rather than
    # integrating over a full optical spectrum.
    # TODO: integrate over spectrum.

    g_pump = modal_gain(n, lambda_pump)
    g_probe = modal_gain(n, lambda_probe)

    E_pump = h * c / lambda_pump
    E_probe = h * c / lambda_probe

    stimulated_pump = (
        g_pump * P_pump / E_pump
    )

    stimulated_probe = (
        g_probe * P_probe / E_probe
    )

    stimulated = (
        stimulated_pump
        + stimulated_probe
    ) / (W * d)

    # Recombination
    recombination = (
        A * n
        + B * n**2
        + C * n**3
    )

    # Electrical injection
    injection = I / (q * W * d * L)

    return injection - recombination - stimulated


# ============================================================
# Initial carrier density
# ============================================================

# Start close to transparency.
# We first calculate a steady state with no pump.

def steady_state_derivative(t, n):
    P_probe = propagate_power(
        probe_input_power,
        n,
        lambda_probe
    )

    g_probe = modal_gain(n, lambda_probe)

    E_probe = h * c / lambda_probe

    stimulated = (
        g_probe * P_probe / E_probe
    ) / (W * d)

    recombination = (
        A * n
        + B * n**2
        + C * n**3
    )

    injection = I / (q * W * d * L)

    return injection - recombination - stimulated


# Initial guess
n_initial = np.full(Nz, 1.5e24)


# Find approximately steady state
steady = solve_ivp(
    steady_state_derivative,
    (0.0, 10e-9),
    n_initial,
    method="BDF",
    rtol=1e-5,
    atol=1e18,
)

n_ss = steady.y[:, -1]


# ============================================================
# Dynamic simulation
# ============================================================

t_start = 0.0
t_end = 2.0e-9

t_eval = np.linspace(
    t_start,
    t_end,
    1000
)

solution = solve_ivp(
    dn_dt,
    (t_start, t_end),
    n_ss,
    t_eval=t_eval,
    method="BDF",
    rtol=1e-5,
    atol=1e18,
)


# ============================================================
# Calculate observable gain versus time
# ============================================================

probe_output = np.empty(len(t_eval))
pump_output = np.empty(len(t_eval))

for j, t in enumerate(t_eval):

    n = solution.y[:, j]

    Pp = propagate_power(
        pump_input_power(t),
        n,
        lambda_pump
    )

    Ps = propagate_power(
        probe_input_power,
        n,
        lambda_probe
    )

    pump_output[j] = Pp[-1]
    probe_output[j] = Ps[-1]


probe_gain = probe_output / probe_input_power

probe_gain_dB = 10.0 * np.log10(probe_gain)


# ============================================================
# Plot carrier density
# ============================================================

plt.figure()

plt.imshow(
    solution.y,
    aspect="auto",
    origin="lower",
    extent=[
        t_eval[0] * 1e12,
        t_eval[-1] * 1e12,
        0,
        L * 1e6,
    ],
)

plt.xlabel("Time (ps)")
plt.ylabel("Position z (um)")
plt.colorbar(label="Carrier density (m$^{-3}$)")

plt.title("SOA carrier-density dynamics")


# ============================================================
# Plot pump
# ============================================================

plt.figure()

plt.plot(
    t_eval * 1e12,
    pump_output * 1e3,
)

plt.xlabel("Time (ps)")
plt.ylabel("Pump output power (mW)")

plt.title("Pump output")


# ============================================================
# Plot probe gain
# ============================================================

plt.figure()

plt.plot(
    t_eval * 1e12,
    probe_gain_dB,
)

plt.xlabel("Time (ps)")
plt.ylabel("Probe gain (dB)")

plt.title("SOA gain depletion and recovery")

plt.grid()

plt.show()