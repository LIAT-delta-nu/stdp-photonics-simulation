DEBUG = True

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from collections.abc import Callable
from config import toml_config
import copy

if DEBUG:
    import time

import devices.SOA_model_Li_2008 as soa
import signals.signals as signals
from spectrum.spectrum import DiscreteSpectrum
from plots import plot_carrier_density, plot_probe_gain_dB, plot_signal_output

CASES = (
    "case_a",
    # "case_b", 
    # "case_c", 
    # "case_d",
)

@toml_config(section="soa_sim_params.single_pulse")
@dataclass
class SinglePulseSoaGainSimulationParams:
    # Spatial and spectral discretization
    Nz: int = 100
    Nlambda: int = 101

    # Simulation time
    t_start: float = 0.0
    t_end: float = 3e-9

    # Gain spectrum
    spectrum_wavelength_min: float = 1500e-9    #TODO: add better default value
    spectrum_wavelength_max: float = 1600e-9    #TODO: add better default value

    # Pump
    pump_wavelength: float = 1551e-9
    pump_power_peak: float = 5e-3
    pump_pulse_width: float = 65e-12
    pump_t_peak: float = 200e-12

    # Probe
    probe_wavelength: float = 1580e-9
    probe_power: float = 40e-9

    # Simulation initial condition
    n_initial : float | np.ndarray = 6.6e24


class SoaGainSimulation():
    soa_conditions: soa.SoaConditions
    soa_material: soa.SoaMaterial
    soa_dims: soa.SoaDimensions
    z: np.ndarray
    t_span: tuple
    input_signals_forward: Callable[[float], float]
    input_signals_backward: Callable[[float], float]
    spectrum: DiscreteSpectrum
    n_initial: float | np.ndarray

    def __init__(
        self,
        sim_params: SinglePulseSoaGainSimulationParams,
        soa_conditions: soa.SoaConditions,
        soa_material: soa.SoaMaterial,
        soa_dims: soa.SoaDimensions,
    ):

        # Soa description
        self.soa_conditions = soa_conditions
        self.soa_material = soa_material
        self.soa_dims = soa_dims

        # Spatial and spectral discretization:
        self.update_z(soa_dims.L, sim_params.Nz)
        self.update_spectrum(
            sim_params.spectrum_wavelength_min,
            sim_params.spectrum_wavelength_max,
            sim_params.Nlambda,
        )

        # Pump and probe input signals:
        ## Pump (forward)
        pump_signal = self.input_signal_spectrum_wrapper(
            signal_power=lambda t : signals.gaussian_pulse(
                t,
                power_peak=sim_params.pump_power_peak,
                pulse_width=sim_params.pump_pulse_width,
                t_peak=sim_params.pump_t_peak,
            ),
            wavelength=sim_params.pump_wavelength
        )
        self.input_signals_forward = pump_signal
        
        ## Probe (backwards)
        probe_signal = self.input_signal_spectrum_wrapper(
            signal_power=lambda _ : sim_params.probe_power,
            wavelength=sim_params.pump_wavelength
        )
        self.input_signals_backward = probe_signal

        # Simulation time span
        self.t_span = (sim_params.t_start, sim_params.t_end)
        
        # Simulation initial condition
        if isinstance(sim_params.n_initial,(float, int)):
            self.n_initial = np.full_like(self.z, sim_params.n_initial)
        elif isinstance(sim_params.n_initial,(np.ndarray,)):
            self.n_initial = np.array(sim_params.n_initial)
        else:
            print("Warning: n_initial for SOA propagation simulation is of unknown type. Using default 6.6e24")
            self.n_initial = np.full_like(self.z, 6.6e24)

    def update_z(self, L, Nz):
        self.z = np.linspace(0.0, L, Nz)

    def update_spectrum(
        self,
        spectrum_wavelength_min,
        spectrum_wavelength_max,
        Nlambda
    ):
        # spectrum
        self.spectrum = DiscreteSpectrum(
            min=spectrum_wavelength_min, 
            max=spectrum_wavelength_max, 
            N=Nlambda
        )

    def clear_input_signal_forward(self):
        self.input_signals_forward = lambda _: 0

    def clear_input_signal_backward(self):
        self.input_signals_backward = lambda _: 0

    def input_signal_spectrum_wrapper(
        self,
        signal_power : Callable[[float], float],
        wavelength : float,
    ) -> Callable[[float], float]:
        signal = input_signal_spectrum_wrapper(
            signal_power=signal_power,
            spectrum=self.spectrum,
            wavelength=wavelength
        )
        return signal

    def run(self):
        if DEBUG:
            start_1 = time.perf_counter()

        solution = solve_ivp(
            fun=soa.dn_dt,
            t_span=self.t_span,
            y0=self.n_initial,
            args=(
                self.soa_conditions,
                self.soa_material,
                self.soa_dims,
                self.input_signals_forward,
                self.input_signals_backward,
                self.spectrum,
                self.z,
            ),
            method="BDF",
            rtol=1e-10,
            atol=1e0,
        )
        if DEBUG:
            stop_1 = time.perf_counter()
            elapsed_1 = stop_1 - start_1
            print(f"solve_ivp execution time: {elapsed_1} seconds.")
        return solution


def input_signal_spectrum_wrapper(
    signal_power : Callable[[float], float],
    spectrum : DiscreteSpectrum,
    wavelength : float,
) -> Callable[[float], float]:
    i = round((wavelength - spectrum.min) / spectrum.width())
    if i < 0 or i >= spectrum.N:
        print(f"Warning in input_signal_spectrum_wrapper: input signal wavelength is outside the simulated spectrum (wavelength={wavelength}, spectrum wavelength range: [{spectrum.min} - {spectrum.max}]). Input signal power will be 0 across whole spectrum.")
        return lambda t: np.zeros_likes(spectrum.spectrum)
    return lambda t: np.eye(1, spectrum.N, i).ravel() * signal_power(t)


# def combine_input_signals(
#     input_signals: list[Callable[[float], np.ndarray]]
# ):
#     return lambda t: np.sum(np.array([signal(t) for signal in input_signals]), axis=0)


def main():

    default_soa_conditions=soa.SoaConditions.from_toml("./sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_conditions")
    default_soa_material=soa.SoaMaterial.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_material")
    default_soa_dims=soa.SoaDimensions.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_dimensions")
    default_sim_params = SinglePulseSoaGainSimulationParams.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_sim_params")
    
    for case in CASES:
        if DEBUG:
            print(case)

        # First, simulate SOA propagation of only the probe to find the initial condition of the main simulation
        # Second, do the main simulation (pump and probe) using use the steady state of the probe-only simulation 
        # as initial n(z) 

        soa_conditions = copy.copy(default_soa_conditions)
        soa_material = copy.copy(default_soa_material)
        soa_dims = copy.copy(default_soa_dims)
        sim_params =  copy.copy(default_sim_params)

        soa_conditions.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_conditions")
        soa_material.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_material")
        soa_dims.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_dimensions")
        sim_params.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_sim_params")

        #########################################
        # Simulation to find initial condition  #
        #########################################

        sim_params_initial_condition = copy.copy(sim_params)
        sim_params_initial_condition.t_start = 0
        sim_params_initial_condition.t_end = 3e-3
        sim_params_initial_condition.pump_power_peak = 0.0

        sim_initial_condition = SoaGainSimulation(
            sim_params=sim_params_initial_condition,
            soa_conditions=soa_conditions,
            soa_material=soa_material,
            soa_dims=soa_dims,
        )

        ## Simulate SOA propagation with probe and without pump to find steady state n(z)
        solution_sim_initial_condition = sim_initial_condition.run()

        #########################################
        # Main simulation                       #
        #########################################

        sim_params_main = copy.copy(sim_params)
        
        # Use steady state n(z) (with pump and without probe) as initial condition for main simulation
        sim_params_main.n_initial = solution_sim_initial_condition.y[:, -1]

        sim_main = SoaGainSimulation(
            sim_params=sim_params_main,
            soa_conditions=soa_conditions,
            soa_material=soa_material,
            soa_dims=soa_dims,
        )
        
        # Run main simulation
        solution_sim_main = sim_main.run()

        #########################################
        # Processing simulation results         #
        #########################################

        # Calc. probe and pump output
        t_eval = solution_sim_main.t

        if DEBUG:
            start_probe = time.perf_counter()

        probe_signal = sim_main.input_signals_backward # probe is propagated backwards
        pump_signal = sim_main.input_signals_forward # pump is propagated forward
        z = sim_main.z

        probe_output = np.sum(
            [
                soa.propagate_power(
                    probe_signal(t), 
                    sim_main.spectrum,
                    n,
                    z,
                    soa_material,
                    soa_dims,
                    forward=False,
                )[0]    # probe is backwards propagated so probe output is at [0]
                for n, t in zip(solution_sim_main.y.T, t_eval)
            ],
            axis=1
        )

        if DEBUG:
            stop_probe = time.perf_counter()
            elapsed_probe = stop_probe - start_probe
            print(f"Probe power propagation execution time: {elapsed_probe} seconds.")
            start_pump = time.perf_counter()

        pump_output = np.sum(
            [
                soa.propagate_power(
                    pump_signal(t),  # pump is propagated forward
                    sim_main.spectrum,
                    n,
                    z,
                    soa_material,
                    soa_dims,
                    forward=True,
                )[-1]   # pump is forwards propagated so pump output is at [-1]
                for n, t in zip(solution_sim_main.y.T, t_eval)
            ],
            axis=1
        )

        if DEBUG:
            stop_pump = time.perf_counter()
            elapsed_pump = stop_pump - start_pump
            print(f"Pump power propagation execution time: {elapsed_pump} seconds.")

        # Probe gain
        probe_gain_dB = [10*np.log10(p/np.sum(probe_signal(t))) for p, t in zip(probe_output, t_eval)]

        # TODO: calc pump pulse input and output total energy, takinginto account that 
        # t_eval is not evenly spaced
        # 
        # pump_input_energy = 
        # pump_output_energy =

        # ============================================================
        # Plots
        # ============================================================

        plot_carrier_density(
            t_eval,
            z,
            solution_sim_main.y
        )
        plot_probe_gain_dB(
            t_eval, 
            probe_gain_dB
        )

        plot_signal_output(t_eval, pump_output)

        # ============================================================
        # Plot pump 
        # ============================================================


        # pump_output[j] = pump_propagation[-1]
        # probe_output[j] = probe_propagation[0]

        # for j, t in enumerate(t_eval):

        #     n = solution.y[:, j]

        #     pump_propagation = soa.propagate_power(
        #         pump_signal.power(t),
        #         n,
        #         pump_signal.lamda,
        #         sim.z(),
        #         sim.material,
        #         sim.dims,
        #         forward=True,
        #     )

        #     probe_propagation = soa.propagate_power(
        #         probe_signal.power(t),
        #         n,
        #         probe_signal.lamda,
        #         sim.z(),
        #         sim.material,
        #         sim.dims,
        #         forward=False,
        #     )

        #     pump_output[j] = pump_propagation[-1]
        #     probe_output[j] = probe_propagation[0]

        #     pump_gain[j] = ( pump_output[j]) / ( pump_signal.power(t) + 1e-50 )



        # # Pump output
        # plt.figure()

        # plt.plot(
        #     t_eval * 1e12,
        #     pump_output * 1e3,
        # )

        # plt.xlabel("Time (ps)")
        # plt.ylabel("Pump output power (mW)")

        # plt.title("Pump output")

        # # Pump input and output power (normalized)
        
        # pump_input = np.array([pump_signal.power(t) for t in t_eval])
        # pump_input_max = pump_power_peak
        # pump_output_max = np.max(pump_output)

        # plt.figure()
        # plt.plot(
        #     t_eval * 1e12,
        #     pump_input/pump_input_max,
        #     label="Input"
        # )
        # plt.plot(
        #     t_eval * 1e12,
        #     pump_output/pump_output_max,
        #     label="Output"
        # )

        # plt.legend()

        # plt.xlabel("Time (ps)")
        # plt.ylabel("Pump power (a.u.)")

        # plt.title("Pump power")


        # ============================================================
        # Plot probe gain
        # ============================================================

        # plot_probe_gain_dB(t_eval, )

        # # ============================================================
        # # Plot pump gain
        # # ============================================================

        # plt.figure()

        # plt.plot(
        #     t_eval * 1e12,
        #     pump_gain_dB,
        # )

        # plt.xlabel("Time (ps)")
        # plt.ylabel("Pump gain (dB)")

        # plt.title(f"SOA gain depletion and recovery (pump at {lambda_pump*1e9:5.1f} nm)")

        # plt.grid()
        # # ============================================================
        # # Plot pump and probe gain
        # # ============================================================

        # plt.figure()

        # plt.plot(
        #     t_eval * 1e12,
        #     probe_gain_dB,
        #     label="probe",
        # )

        # plt.plot(
        #     t_eval * 1e12,
        #     pump_gain_dB,
        #     label="pump",
        # )

        # plt.legend()
        # plt.xlabel("Time (ps)")
        # plt.ylabel("Gain (dB)")

        # plt.title(f"SOA gain depletion and recovery (probe at {lambda_probe*1e9:5.1f} nm and pump at {lambda_pump*1e9:5.1f} nm)")

        # plt.grid()

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()