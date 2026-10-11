import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections.abc import Callable
from dataclasses import dataclass, astuple

# ============================================================
# Modal gain calc
# ============================================================
def _modal_gain(
    n, 
    lamda,
    n0, 
    lambda_0, 
    a0, 
    a1, 
    a2
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
        - a1 * (lamda - lambda_n) ** 2
    )

# ============================================================
# Recombination rates
# ============================================================

def R_defect(
    n,
    A = 2.0e8,  # 1/s
):
    return A * n

def R_spontaneous_radiative(
    n,
    B = 3.83e-17    # m^3/s
):
    return B * n**2

def R_auger(
    n,
    C = 4.0e-41 # m^6/s
):
    return C * n**3

def R_tot(
    n,
    A = 2.0e8,      # 1/s
    B = 3.83e-17,   # m^3/s
    C = 4.0e-41,    # m^6/s
):
    R_tot = (
        R_defect(n,A)
        + R_spontaneous_radiative(n,B)
        + R_auger(n, C)
    )
    return R_tot

# ============================================================
# SOA description dataclasses
# ============================================================

@dataclass()
class SoaCondition:
    I: float = 87e-3


@dataclass()
class SoaMaterial:
    # Transparency
    n0: float = 1.0e24       # Transparency carrier concentration, m^-3
    lambda_0: float = 1.585e-6  # Transparency wavelength, m

    # Modal gain parameters
    a0: float = 4.5e-21      # Modal gain parameter, m^2
    a1: float = 1.5e18       # Modal gain parameter, m^-3
    a2: float = 5.0e-33      # Modal gain parameter, m^4

    # Carrier recombination coefficients
    A: float = 2.0e8         # Defect recombination, s^-1
    B: float = 3.83e-17      # Spontaneous radiative recombination, m^3/s
    C: float = 4.0e-41       # Auger recombination, m^6/s

    # Loss and spontaneous emission
    alpha: float = 2e3       # Intrinsic material loss, m^-1
    beta_eff: float = 7.3e-3 # Effective spontaneous emission factor


    def modal_gain(self, n: float, lamda: float):
        return _modal_gain(n, lamda, self.n0, self.lambda_0, self.a0, self.a1, self.a2)


    def R_defect(self, n):
        return R_defect(n, self.A)


    def R_spontaneous_radiative(self, n):
        return R_spontaneous_radiative(n, self.B)


    def R_auger(self, n):
        return R_auger(n, self.C)


    def R_tot(self, n):
        return R_tot(n, self.A, self.B, self.C)


@dataclass()
class SoaDimensions:
    W: float    # Active region width
    d: float    # Active region hickness
    L: float    # Active region length

    def area(self):
        return self.W * self.d

    def volume(self):
        return self.W * self.d * self.L

# ============================================================
# SOA description dataclasses
# ============================================================

# TODO: esto es muy feo y tendria que volar. Creo que estamal asumir que una entrada tiene siempre la misma long. de onda
@dataclass
class InputSignal:
    lamda: float
    power: Callable[[float,], float]

# ============================================================
# Physical constants
# ============================================================

q = 1.602176634e-19       # C
h = 6.62607015e-34        # J s
c = 299792458.0           # m/s


# ============================================================
# Simulation parameters
# ============================================================

# I = 72e-3                 # 72 mA
# # I = 0 #72e-3                 # 72 mA



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

# Weak CW probe
probe_input_power = 40e-6     # W


# ============================================================
# Optical propagation (based on Li et al 2008)
# ============================================================

# NOTE: asume que hay una única wavelength. 
# Si quiero propagar señales que sean superposicion de varias
# longitudes de onda, esta funcion no me lo permite.

def propagate_power(
    P_in, 
    n, 
    wavelength, 
    z,
    material : SoaMaterial,
    dims : SoaDimensions,
    # TODO: recibir params del SOA por argumento y no leyendo vars. globales
    # B = 3.83e-17,    # m^3/s
    # alpha = 2e3,      # m^-1
    # beta_eff = 7.3e-3,
):
    """
    Propagate a forward travelling optical signal
    through the SOA for the current carrier-density profile.
    """

    # Ec. dif. de la forma: y' + f0 * y = g0, con f0 y g0 ctes.
    # Solucion:
    # y = [y(0) - g0/f0 ] * exp(-f0*z) + g0/f0

    # y = P(z)
    # f0 = -(gm-alpha)   (asumido constante para dz suficientemente chico)
    # g0 = W*d*B*n^2*(hc/lambda)beta_eff/2  (asumido constante para dz suficientemente chico)


    P = np.zeros_like(z)
    P[0] = P_in
    dz = (z[-1] - z[0]) / (len(z) - 1)

    for i in range(len(z) - 1):

        gm = material.modal_gain(n[i], wavelength)    # TODO: esto no funciona si se ensancha un poco el espectro

        f0 = -(gm - material.alpha)
        g0 = material.beta_eff/2 * dims.area()*material.B*n[i]*n[i]*h*c/wavelength    # TODO: y si tengo un pulso con espectro ancho? que hago con eso aca

        P[i + 1] = (P[i] - g0/f0) * np.exp(-f0 * dz) + g0/f0

    return P


# ============================================================
# Carrier-density dynamics
# ============================================================

def dn_dt(
    t, 
    n,
    condition: SoaCondition,
    material: SoaMaterial,
    dims: SoaDimensions,
    input_signals: list[InputSignal],
    z: np.ndarray
):
    """
    dn/dt for every longitudinal SOA section.
    """
    ###
    # dn/dt = injection - recombination - stimulated
    ###

    # Electrical injection
    injection = condition.I / (q * dims.volume())

    # Recombination
    recombination = material.R_tot(n)

    # Stimulated emission and absorption contribution
    # For the present simplified model we use the
    # pump and probe powers directly rather than
    # integrating over a full optical spectrum.
    # TODO: integrate over spectrum.
    stimulated = np.zeros_like(n)
    for input_signal in input_signals:
        gm = material.modal_gain(n, input_signal.lamda)
        E_photon = h * c / input_signal.lamda
        stimulated += propagate_power(
            input_signal.power(t),
            n,
            input_signal.lamda,
            z,
            material,
            dims
        ) * gm / E_photon / dims.area()

    return injection - recombination - stimulated

    # # Optical powers at this instant
    # P_pump = propagate_power(
    #     pump_input_power(t),
    #     n,
    #     lambda_pump,
    #     z
    # )

    # P_probe = propagate_power(
    #     probe_input_power,
    #     n,
    #     lambda_probe,
    #     z
    # )

    # g_pump = material.modal_gain(n, lambda_pump)
    # g_probe = material.modal_gain(n, lambda_probe)

    # E_pump = h * c / lambda_pump
    # E_probe = h * c / lambda_probe

    # stimulated_pump = (
    #     g_pump * P_pump / E_pump
    # )

    # stimulated_probe = (
    #     g_probe * P_probe / E_probe
    # )

    # stimulated = (
    #     stimulated_pump
    #     + stimulated_probe
    # ) / dims.area





# # ============================================================
# # Initial carrier density
# # ============================================================

# # Start close to transparency.
# # We first calculate a steady state with no pump.

# def steady_state_derivative(
#     t, 
#     n,
#     material : SoaMaterial,
#     dims: SoaDimensions,
#     input_signals: list[InputSignal],
#     z: np.ndarray
# ):
#     P_probe = propagate_power(
#         probe_input_power,
#         n,
#         lambda_probe,
#         z
#     )

#     g_probe = material.modal_gain(n, lambda_probe)

#     E_probe = h * c / lambda_probe

#     stimulated = (
#         g_probe * P_probe / E_probe
#     ) / (W * d)

#     recombination = R_tot(n) # TODO: make steady_state_derivative receive parameters for A, B and C and forward them to R_tot

#     injection = I / (q * W * d * L)

#     return injection - recombination - stimulated

def main():

    # SOA dims
    L = 800e-6                # 800 um
    W = 2.0e-6                # active-region width [m]
    d = 0.2e-6                # active-region thickness [m]

    # SOA injection current
    I = 72e-3                 # 72 mA


    # Input signals wavelengths
    lambda_pump = 1551e-9
    lambda_probe = 1550e-9

    pump_power_peak = 5e-3
    pump_pulse_width = 65e-12   # FWHM
    pump_t_peak = 200e-12


    # Simulation params
    Nz = 100
    z = np.linspace(0.0, L, Nz)

    t_start = 0.0
    t_end = 2.0e-9
    t_eval = np.linspace(
        t_start,
        t_end,
        1000
    )

    condition = SoaCondition(I=87e-3)
    material = SoaMaterial()
    dims = SoaDimensions(W, d, L)

    pump_signal : InputSignal = InputSignal(
        lamda = lambda_pump,
        power = lambda t: gaussian_pulse(t, pump_power_peak, pump_pulse_width, pump_t_peak)
    )

    probe_signal : InputSignal = InputSignal(
        lamda = lambda_probe,
        power = lambda t: probe_input_power
    )

    input_signals = (pump_signal, probe_signal,)


    # ============================================================
    # Dynamic simulation
    # ============================================================

    # Initial guess to calculate steady state
    n_initial = np.full(Nz, 2.7e26)

    # Find approximately steady state
    steady = solve_ivp(
        fun=dn_dt,
        t_span=(0.0, 100e-9),
        y0=n_initial,
        args=(
            condition,
            material,
            dims,
            (probe_signal,),
            z
        ),
        method="BDF",
        rtol=1e-5,
        atol=1e18,
    )

    print(steady.message)
    print(steady.y[-1,:])

    # Use steady state as initial condition for main simulation
    n_ss = steady.y[:, -1]
    # print(n_ss)
    solution = solve_ivp(
        fun=dn_dt,
        t_span=(t_start, t_end),
        y0=n_ss,    # start from steady state
        t_eval=t_eval,
        args=(
            condition,
            material,
            dims,
            (pump_signal, probe_signal),
            z
        ),
        method="BDF",
        rtol=1e-5,
        atol=1e18,
    )


    # ============================================================
    # Calculate observable gain versus time
    # ============================================================

    probe_output = np.empty(len(t_eval))
    pump_output = np.empty(len(t_eval))

    pump_gain = np.zeros_like(pump_output)

    for j, t in enumerate(t_eval):

        n = solution.y[:, j]

        Pp = propagate_power(
            pump_signal.power(t),
            n,
            pump_signal.lamda,
            z,
            material,
            dims
        )

        Ps = propagate_power(
            probe_signal.power(t),
            n,
            probe_signal.lamda,
            z,
            material,
            dims
        )

        pump_output[j] = Pp[-1]
        probe_output[j] = Ps[-1]

        pump_gain[j] = ( pump_output[j]) / ( pump_signal.power(t) + 1e-50 )


    probe_gain = probe_output/probe_input_power
    pump_gain_dB = 10.0 * np.log10(pump_gain)
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

    plt.title(f"SOA gain depletion and recovery (probe at {lambda_probe*1e9:5.1f} nm)")

    plt.grid()


    # ============================================================
    # Plot pump gain
    # ============================================================

    plt.figure()

    plt.plot(
        t_eval * 1e12,
        pump_gain_dB,
    )

    plt.xlabel("Time (ps)")
    plt.ylabel("Pump gain (dB)")

    plt.title(f"SOA gain depletion and recovery (pump at {lambda_pump*1e9:5.1f} nm)")

    plt.grid()
    # ============================================================
    # Plot probe gain
    # ============================================================

    plt.figure()

    plt.plot(
        t_eval * 1e12,
        probe_gain_dB,
        label="probe",
    )

    plt.plot(
        t_eval * 1e12,
        pump_gain_dB,
        label="pump",
    )

    plt.legend()
    plt.xlabel("Time (ps)")
    plt.ylabel("Gain (dB)")

    plt.title(f"SOA gain depletion and recovery (probe at {lambda_probe*1e9:5.1f} nm and pump at {lambda_pump*1e9:5.1f} nm)")

    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
    # n = np.linspace(1e9, 1e30)

    # gain_pump = [_modal_gain(dn, lambda_pump) for dn in n]
    # gain_probe = [_modal_gain(dn, lambda_probe) for dn in n]


    # plt.figure()

    # plt.plot(
    #     n,
    #     gain_pump,
    # )

    # # plt.plot(
    # #     delta_n,
    # #     gain_probe,
    # # )

    # plt.xlabel("n")
    # plt.ylabel("modal gain")

    # plt.grid()

    # plt.show()

    # lambdas = np.linspace(1585e-9, 1595e-9)
    
    # n0 = 1.0e24               # m^-3
    # gain_pump = [_modal_gain(n0, lamda) for lamda in lambdas]


    # plt.figure()

    # plt.plot(
    #     lambdas,
    #     gain_pump,
    # )

    # # plt.plot(
    # #     delta_n,
    # #     gain_probe,
    # # )

    # plt.xlabel("lambda")
    # plt.ylabel("modal gain")

    # plt.grid()

    # plt.show()

