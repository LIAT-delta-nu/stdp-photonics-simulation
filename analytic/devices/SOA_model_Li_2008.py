import numpy as np
from collections.abc import Callable
from dataclasses import dataclass, astuple
from config.toml_config import toml_config
from spectrum.spectrum import DiscreteSpectrum

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

@toml_config(section="soa_conditions")
@dataclass
class SoaConditions:
    I: float = 87e-3

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
# Physical constants
# ============================================================

q = 1.602176634e-19       # C
h = 6.62607015e-34        # J s
c = 299792458.0           # m/s


# ============================================================
# Optical propagation (based on Li et al 2008)
# ============================================================

# P_in [np.ndarray]: 
#   arreglo con la potencia por bin espectral
#
# spectrum [np.ndarray]: 
#   arreglo con la frecuencia central de cada bin espectral
#
#  z [np.ndarray]: 
#   posiciones dentro del SOA. Se asume que estan uniformemente 
#   distribuidas.

def propagate_power(
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


# ============================================================
# Carrier-density dynamics
# ============================================================

def dn_dt(
    t, 
    n,
    condition: SoaConditions,
    material: SoaMaterial,
    dims: SoaDimensions,
    input_signal_forward : Callable[[float], np.ndarray],
    input_signal_backward : Callable[[float], np.ndarray],
    spectrum: DiscreteSpectrum,
    z: np.ndarray,
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

    # TODO: optionally, to speed up simulation (how much?), gm(n, lambda) 
    # can be calculated in this function and be passed to the propagate 
    # power function, so that it is not calculated again.
    gm = np.array([material.modal_gain(n_i, spectrum.spectrum) for n_i in n] )
    P_forward = propagate_power(
        input_signal_forward(t),
        spectrum,
        n,
        z,
        material,
        dims,
        forward=True,
    )
    P_backward = propagate_power(
        input_signal_backward(t),
        spectrum,
        n,
        z,
        material,
        dims,
        forward=False,
    )
    P = P_forward + P_backward  # shape (n_z, n_spectrum), so indexing is P[z, lambda]
    E_photon = h * c / spectrum.spectrum

    stimulated  = np.sum(gm * P / E_photon, axis=1) / dims.area()

    return injection - recombination - stimulated


if __name__ == '__main__':
    pass