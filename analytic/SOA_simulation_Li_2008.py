DEBUG = True
OUTDIR = "./out"


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
if DEBUG:
    import time

from config import toml_config
from config import cli_config
import copy
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

@cli_config
@toml_config(section="soa_sim_params")
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
    i = round((wavelength - spectrum.min) * spectrum.N / spectrum.width())
    if i < 0 or i >= spectrum.N:
        print(f"Warning in input_signal_spectrum_wrapper: input signal wavelength is outside the simulated spectrum (wavelength={wavelength}, spectrum wavelength range: [{spectrum.min} - {spectrum.max}]). Input signal power will be 0 across whole spectrum.")
        return lambda t: np.zeros_likes(spectrum.spectrum)
    return lambda t: np.eye(1, spectrum.N, i).ravel() * signal_power(t)


# def combine_input_signals(
#     input_signals: list[Callable[[float], np.ndarray]]
# ):
#     return lambda t: np.sum(np.array([signal(t) for signal in input_signals]), axis=0)


def main():

    parser = ArgumentParser()
    SinglePulseSoaGainSimulationParams.add_cli_args_to_parser(parser)
    args = parser.parse_args()

    default_soa_conditions=soa.SoaConditions.from_toml("./sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_conditions")
    default_soa_material=soa.SoaMaterial.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_material")
    default_soa_dims=soa.SoaDimensions.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_dimensions")
    default_sim_params = SinglePulseSoaGainSimulationParams.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_sim_params")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    default_sim_params.update_from_cli_args(args)

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
        sim_params.update_from_cli_args(args)

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
        n = solution_sim_main.y
        t_eval = solution_sim_main.t

        if DEBUG:
            start_probe = time.perf_counter()

        probe_signal = sim_main.input_signals_backward # probe is propagated backwards
        pump_signal = sim_main.input_signals_forward # pump is propagated forward
        z = sim_main.z

        probe_output_power = np.sum(
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
                for n, t in zip(n.T, t_eval)
            ],
            axis=1
        )

        if DEBUG:
            stop_probe = time.perf_counter()
            elapsed_probe = stop_probe - start_probe
            print(f"Probe power propagation execution time: {elapsed_probe} seconds.")
            start_pump = time.perf_counter()

        pump_output_power = np.sum(
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
                for n, t in zip(n.T, t_eval)
            ],
            axis=1
        )

        if DEBUG:
            stop_pump = time.perf_counter()
            elapsed_pump = stop_pump - start_pump
            print(f"Pump power propagation execution time: {elapsed_pump} seconds.")


        # Probe input power and gain
        probe_input_power = np.array(np.sum([probe_signal(t) for t in t_eval], axis=1)) # sum across spectrum to get total power
        probe_gain_dB = np.array([10*np.log10(p_out/p_in) for p_out, p_in in zip(probe_output_power, probe_input_power)])

        # Pump input power
        pump_input_power = np.array(np.sum([pump_signal(t) for t in t_eval], axis=1)) # sum across spectrum to get total power

        # TODO: calc pump pulse input and output total energy, takinginto account that 
        # t_eval is not evenly spaced
        # 
        # pump_input_energy = 
        # pump_output_energy =

        # ============================================================
        # Save configs
        # ============================================================
        save_dir = Path(OUTDIR) / timestamp / str(case)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_outfile = save_dir / "config.toml"
        soa_conditions.to_toml(config_outfile)
        soa_material.to_toml(config_outfile)
        soa_dims.to_toml(config_outfile)
        sim_params.to_toml(config_outfile)

        # ============================================================
        # Save results
        # ============================================================

        np.savez(
            save_dir / "n_sim_result.npz",
            t=t_eval,
            z=z,
            n=n,
        )
        np.savez(
            save_dir / "pump_and_probe_powers_and_gain.csv",
                t=t_eval,
                pump_input_power_W=pump_input_power,
                pump_output_power_W=pump_output_power,
                probe_input_power_W=probe_input_power,
                probe_output_power_W=probe_output_power,
                probe_gain_dB=probe_gain_dB,
        )
        np.savetxt(
            save_dir / "n.csv", 
            n,
            delimiter=',',
        )
        data = np.block([
                arr[:, np.newaxis]
                for arr in (
                    t_eval,
                    pump_input_power,
                    pump_output_power,
                    probe_input_power,
                    probe_output_power,
                    probe_gain_dB,
                )
            ]
        )
        np.savetxt(
            save_dir / "pump_and_probe_powers_and_gain.csv",
            data,
            comments='',
            delimiter=',',
            header="t[s], Pump input power [W], Pump output power [W], Probe input power [W], Probe output power [W], Probe gain [dB]"
        )

        # ============================================================
        # Plots
        # ============================================================
        ### Carrier desity ###
        plot_carrier_density(
            t_eval,
            z,
            n,
            outfile=save_dir / "carrier_density.svg",
        )

        ### Probe gain dB ###
        plot_probe_gain_dB(
            t_eval, 
            probe_gain_dB=probe_gain_dB,
            pump_output=pump_output_power,
            show=False,
            outfile=save_dir / "probe_gain_dB.svg",
        )

        ### Pump input and output ###
        _, ax = plot_signal_output(
            t_eval,
            pump_input_power,
            label="Input",
            # outfile=save_dir / "pump_input_output.svg",
            show=False,
        )

        plot_signal_output(
            t_eval,
            pump_output_power,
            ax=ax,
            label="Output",
            title="Pump signal",
            outfile=save_dir / "pump_input_output.svg",
            show=False,
        )

        ### Pump input and output (normalized) ###
        _, ax = plot_signal_output(
            t_eval,
            pump_input_power/max(pump_input_power),
            label="Input",
            title="Pump signal",
            # outfile=save_dir / "pump_input_output.svg",
            show=False,
        )
        plot_signal_output(
            t_eval,
            pump_output_power/max(pump_output_power),
            ax=ax,
            label="Output",
            ylabel="Normalized power [a.u.]",
            outfile=save_dir / "pump_input_output_normalized.svg",
            show=False,
        )


    plt.show()

if __name__ == '__main__':
    main()