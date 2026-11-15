DEBUG = True

from abc import ABC, abstractmethod
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass, astuple
from config.toml_config import toml_config
from config.cli_config import cli_config
from spectrum.spectrum import DiscreteSpectrum
from scipy.integrate import solve_ivp
import time

# ============================================================
# Physical constants
# ============================================================

q = 1.602176634e-19       # C
h = 6.62607015e-34        # J s
c = 299792458.0           # m/s

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

def _R_defect(
    n,
    A = 2.0e8,  # 1/s
):
    return A * n

def _R_spontaneous_radiative(
    n,
    B = 3.83e-17    # m^3/s
):
    return B * n**2

def _R_auger(
    n,
    C = 4.0e-41 # m^6/s
):
    return C * n**3

def _R_tot(
    n,
    A = 2.0e8,      # 1/s
    B = 3.83e-17,   # m^3/s
    C = 4.0e-41,    # m^6/s
):
    R_tot = (
        _R_defect(n,A)
        + _R_spontaneous_radiative(n,B)
        + _R_auger(n, C)
    )
    return R_tot

# ============================================================
# SOA description dataclasses
# ============================================================
@cli_config
@toml_config(section="soa_conditions")
@dataclass
class SoaConditions:
    I: float = 87e-3

@cli_config
@toml_config(section="soa_material")
@dataclass()
class SoaMaterial:
    # Transparency
    n0: float = 1.0e24       # Transparency carrier concentration, m^-3
    lambda_0: float = 1.585e-6  # Transparency wavelength, m     # DEPENDS ON SOA LENGTH!!!!

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
        return _R_defect(n, self.A)


    def R_spontaneous_radiative(self, n):
        return _R_spontaneous_radiative(n, self.B)


    def R_auger(self, n):
        return _R_auger(n, self.C)


    def R_tot(self, n):
        return _R_tot(n, self.A, self.B, self.C)

@cli_config
@toml_config(section="soa_dimensions")
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
# SOA simulation classes
# ============================================================

@dataclass
class SoaSimulationParams(ABC):
    sim_type_name : str = "Generic SOA simulation"

    # Spatial and spectral discretization
    Nz: int = 101
    Nlambda: int = 101

    # Simulation time
    t_start: float = 0.0
    t_end: float = 3e-9

    # Gain spectrum
    spectrum_wavelength_min: float = 1500e-9    #TODO: add better default value
    spectrum_wavelength_max: float = 1700e-9    #TODO: add better default value

    # Simulation initial condition
    n_initial : float | np.ndarray = 6.6e24
    
    @property
    @abstractmethod
    def input_signals_forward_list(self) -> list[Callable[[float], np.ndarray]]:
        pass

    @property
    @abstractmethod
    def input_signals_backward_list(self) -> list[Callable[[float], np.ndarray]]:
        pass

    @property
    def spectrum(self):
        return DiscreteSpectrum(
            self.spectrum_wavelength_min,
            self.spectrum_wavelength_max,
            self.Nlambda,
        )


@dataclass(init=False)
class SoaSimulation():
    conditions: SoaConditions
    material: SoaMaterial
    dims: SoaDimensions
    sim_type_name: str
    z: np.ndarray
    t_span: tuple
    input_signals_forward_list: list[Callable[[float], float]]
    input_signals_backward_list: list[Callable[[float], float]]
    spectrum: DiscreteSpectrum
    n_initial: float | np.ndarray

    def __init__(
        self,
        soa_conditions: SoaConditions,
        soa_material: SoaMaterial,
        soa_dims: SoaDimensions,
        sim_params : SoaSimulationParams,
    ):
        # Simulation time span
        self.t_span = (sim_params.t_start, sim_params.t_end)

        # Spatial and spectral discretization:
        self.z = np.linspace(0.0, soa_dims.L, sim_params.Nz)
        self.spectrum = sim_params.spectrum

        # Simulation initial condition
        if isinstance(sim_params.n_initial,(float, int)):
            self.n_initial = np.full_like(self.z, sim_params.n_initial)
        elif isinstance(sim_params.n_initial,(np.ndarray,)):
            self.n_initial = np.array(sim_params.n_initial)
        else:
            print("Warning: n_initial for SOA propagation simulation is of unknown type. Using default 6.6e24")
            self.n_initial = np.full_like(self.z, 6.6e24)

        self.conditions=soa_conditions
        self.material=soa_material
        self.dims=soa_dims

        # Note: a local variable is used to force lambda to bind to Nlambda by value and not to self by reference
        # Therefore, if sim_params.Nlambda changes, the lambda stays the same
        Nlambda = sim_params.Nlambda

        no_signal = lambda _: np.zeros(Nlambda)

        # Initialize signal list with at least one signal (which is 0 for the full spectrum)
        # so that input_signals_forward() and input_signals_backward() always return
        # an array
        self.input_signals_forward_list = [no_signal] + sim_params.input_signals_forward_list.copy()
        self.input_signals_backward_list = [no_signal] + sim_params.input_signals_backward_list.copy()

    def run(self):
        if DEBUG:
            start_1 = time.perf_counter()
        solution = solve_ivp(
            fun=self.dn_dt,
            t_span=self.t_span,
            y0=self.n_initial,
            method="BDF",
            rtol=1e-10,
            atol=1e0,
        )
        if DEBUG:
            stop_1 = time.perf_counter()
            elapsed_1 = stop_1 - start_1
            print(f"solve_ivp execution time: {elapsed_1} seconds.")
        return solution

    def get_powers_from_solution(
            self,
            n : np.ndarray, 
            t_eval : np.ndarray,
        ) -> tuple[np.ndarray]:

        if DEBUG:
            start = time.perf_counter()

        power_backward_by_wavelength = np.array([
            self.propagate_power(t, n_, forward=False)
            for n_, t in zip(n.T, t_eval)
        ])   #[t, z, wavelength]

        if DEBUG:
            stop = time.perf_counter()
            elapsed = stop - start
            print(f"Backward propagation execution time: {elapsed} seconds.")
            start = time.perf_counter()

        power_forward_by_wavelength = np.array([
            self.propagate_power(t, n_, forward=True)
            for n_, t in zip(n.T, t_eval)
        ])   #[t, z, wavelength]

        if DEBUG:
            stop = time.perf_counter()
            elapsed = stop - start
            print(f"Forward propagation execution time: {elapsed} seconds.")

        return power_forward_by_wavelength, power_backward_by_wavelength, power_forward_by_wavelength+power_backward_by_wavelength

    # ============================================================
    # Carrier-density dynamics (based on Li et al 2008)
    # ============================================================

    def dn_dt(self, t, n):
        """
        dn/dt for every longitudinal SOA section.
        """
        ###
        # dn/dt = injection - recombination - stimulated
        ###

        # Electrical injection
        injection = self.conditions.I / (q * self.dims.volume())

        # Recombination
        recombination = self.material.R_tot(n)

        # Stimulated emission and absorption contribution

        # TODO: optionally, to speed up simulation (how much?), gm(n, lambda) 
        # can be calculated in this function and be passed to the propagate 
        # power function, so that it is not calculated again.
        gm = np.array([self.material.modal_gain(n_i, self.spectrum.spectrum) for n_i in n] )
        P_forward = self.propagate_power(
            t,
            n,
            forward=True,
        )
        P_backward = self.propagate_power(
            t,
            n,
            forward=False,
        )
        P = P_forward + P_backward  # shape (n_z, n_spectrum), so indexing is P[z, lambda]
        E_photon = h * c / self.spectrum.spectrum

        stimulated  = np.sum(gm * P / E_photon, axis=1) / self.dims.area()

        return injection - recombination - stimulated

    # ============================================================
    # Optical propagation (based on Li et al 2008)
    # ============================================================
    
    def propagate_power(self, t : float, n : np.ndarray, forward : bool = True):
        return soa_propagate_power(
            P_in = self.input_signal(t, forward),
            spectrum=self.spectrum,
            n=n,
            z=self.z,
            material=self.material,
            dims=self.dims,
            forward=forward
        )

    # ============================================================
    # Input signals
    # ============================================================

    def input_signal(self, t, forward: bool) -> np.ndarray:
        input_signals_list = self.input_signals_forward_list if forward else self.input_signals_backward_list
        return np.sum(
            np.array([signal(t) for signal in input_signals_list]),
            axis=0
        )

    def input_signal_forward(self, t) -> np.ndarray:
        return self.input_signal(t, forward=True)

    def input_signal_backward(self, t) -> np.ndarray:
        return self.input_signal(t, forward=False)


def soa_propagate_power(
    P_in : np.ndarray, 
    spectrum : DiscreteSpectrum, 
    n : np.ndarray, 
    z : np.ndarray,
    material : SoaMaterial,
    dims : SoaDimensions,
    forward : bool = True,
):
    """
    Propagate a travelling optical signal
    through the SOA for the current carrier-density profile.

    Ec. dif. de la forma: y' + f0 * y = g0, con f0 y g0 ctes.
    Solucion:
    y = [y(0) - g0/f0 ] * exp(-f0*z) + g0/f0

    y = P(z)
    f0 = -(gm-alpha)   (asumido constante para dz suficientemente chico)
    g0 = W*d*B*n^2*(hc/lambda)beta_eff/2/N_spectrum  (asumido constante para dz suficientemente chico)

    P_in : np.ndarray
        arreglo con la potencia por bin espectral
    
    spectrum : np.ndarray
        arreglo con la frecuencia central de cada bin espectral
    
    z : np.ndarray
        posiciones dentro del SOA. Se asume que estan uniformemente 
        distribuidas.

    n : np.ndarray
        carrier density (n[i] is carrier density in position z[i])

    forward : bool
        If True, P_in is propagated from z[0] to z[Nz-1], where Nz=len(z)
        If False, P_in is propagated from z[Nz-1] to z[0].

    returns:
        array P of shape (n_z, n_spectrum), so indexing is P[z, lambda], 
        so P[i, j] is the propagated power in z[i] at wavelength spectrum[j].
    """

    P = np.zeros((len(z), spectrum.N)) # shape: (n_z, n_spectrum), so indexing is P[z, lambda]
    P[0] = P_in
    dz = (z[-1] - z[0]) / (len(z) - 1)

    wavelengths = spectrum.spectrum

    if not forward:
        n = np.flip(n)

    for i, n_i in enumerate(n[:-1]):

        gm = material.modal_gain(n_i, wavelengths)

        f0 = -(gm - material.alpha)
        g0 = material.beta_eff/2 * dims.area()*material.B*n_i*n_i*h*c/wavelengths/spectrum.N

        P[i + 1] = (P[i] - g0/f0) * np.exp(-f0 * dz) + g0/f0

    if not forward:
        P[:] = np.flip(P, axis=0)

    return P






if __name__ == '__main__':
    pass