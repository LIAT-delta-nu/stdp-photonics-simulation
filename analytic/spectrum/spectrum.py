import numpy as np
from collections.abc import Callable
import matplotlib.pyplot as plt
import matplotlib as mpl

def lorentzian(x, x0, FWHM, total_power=1):
    gamma = FWHM/2
    return  total_power * 1 / gamma / np.pi / (1 + ((x-x0)/gamma)**2)

def rectangle(x, x0, width, total_power = 1):
    height = 1/width
    return  total_power * height if abs(x-x0) < width/2 else 0

# Discrete spectrum
# Contains a discrete spectrum of N values in [min, max] (including max),
# with an uniform separation of (max-min)/(N-1)
# It is intentionally ambiguous on the spectrum unit:
# It can be used for frequency, energy, or wavelength.
class DiscreteSpectrum():
    min: float
    max: float
    N: int
    spectrum : np.ndarray

    def __init__(self, min: float, max: float, N: int):
        self.update(min, max, N)

    def update(self, min: float, max: float, N: int):
        self.min = min
        self.max = max
        self.N = N
        self.spectrum = np.linspace(min, max, num=N, endpoint=True)

    def width(self):
        return self.max - self.min

    def __getitem__(self, key):
        return self.spectrum[key]

    def __array__(self, dtype=None, copy=None):
        return self.spectrum

    def __iter__(self):
        return iter(self.spectrum)

    def monochromatic_signal_to_full_spectrum(
        self,
        signal_power : Callable[[float], float],
        wavelength : float,
    ) -> Callable[[float], float]:
        i = round((wavelength - self.min) * self.N / self.width())
        if i < 0 or i >= self.N:
            print(f"Warning: input signal wavelength is outside the simulated spectrum (wavelength={wavelength}, spectrum wavelength range: [{spectrum.min} - {spectrum.max}]). Input signal power will be 0 across whole spectrum.")
            return lambda _: np.zeros(self.N)
        signal = lambda t: np.eye(1, self.N, i).ravel() * signal_power(t)
        return signal

    # Takes a continuous psd function and returns a discrete array
    # of unitless coefficients that are an envelope of the psd
    # in the discrete spectrum and where the sum of the coefficients 
    # equals 1.
    #
    # psd: function or callable object that receives a spectrum position
    # and returns the corresponding power/[spectrum unit]
    #
    def normalized_psd_envelope(self, psd: Callable[[float], float]):
        envelope = np.array([psd(pos) for pos in self.spectrum])
        normalized_envelope = envelope / np.sum(envelope)
        return normalized_envelope

    # Takes a continuous psd function and returns a discrete array
    # of powers P_i where 
    #  - The envelope of the P_i array is the same shape
    # as the psd
    #  - Σ P_i = P_tot
    #  - P_i represents the power contribution of the spectrum position i
    #    to the total por P_tot
    #
    # psd: function or callable object that receives a spectrum position
    # and returns the corresponding power/[spectrum unit]
    #
    # P_tot: Total power (Σ P_i = P_tot) [default: 1]
    def psd_to_power(self, psd: Callable[[float], float], P_tot: float = 1):
        return self.normalized_psd_envelope(psd) * P_tot * self.N

if __name__ == '__main__':

    # Test the creation of normalized psd envelope coefficients for the spectrum 
    lambda_min = 1500
    lambda_max = 1600

    total_power = 10

    Ns = (51, 11, 3)

    psd_lorentzian = lambda wavelength : lorentzian(wavelength, 1550, 10, 10)
    psd_rect = lambda wavelength : rectangle(wavelength, 1530, 50, 20)

    for psd, psd_name in zip((psd_rect, psd_lorentzian), ("Rectangular psd", "Lorentzian psd")):
        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

        fine_spectrum = DiscreteSpectrum(lambda_min, lambda_max, 100001)
        psd_values_fine_spectrum = [psd(x) for x in fine_spectrum]
        
        fig, ax = plt.subplots(figsize=(5,3), constrained_layout=True)

        print('--------------------------')
        print(psd_name)
        print()
        for i, N in enumerate(Ns):
            spectrum = DiscreteSpectrum(lambda_min, lambda_max, N)
            envelope_coeffs = spectrum.normalized_psd_envelope(psd)

            markerline, stemline, baseline = ax.stem(
                spectrum, 
                envelope_coeffs * total_power, 
                markerfmt="^", 
                linefmt=colors[i],
                basefmt='none',
                label=f"N={N}"
            )
            stemline.set_color(colors[i])
            ax.set_ylabel("Power spectral density [W/nm]")
            ax.set_xlabel("Wavelength [nm]")
            ax.grid(True)
            print(f"N = {N}")
            print(f"Envelope coefficients sum: {np.sum(envelope_coeffs)}")
            print()

        print()
        ax.plot(fine_spectrum, psd_values_fine_spectrum, color='k', label='PSD', lw=2.0)
        ax.legend()

    plt.show()

