VERBOSE = True

import numpy as np
from dataclasses import dataclass, field
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path
from argparse import ArgumentParser
import copy

from stdp_photonics.experiment_setup import ExperimentSetup, add_args_to_parser, load_simulation_results_from_dir
from stdp_photonics.spectrum import DiscreteSpectrum
from stdp_photonics.results_manager import (
    save_configs,
    save_soa_sim_result,
    save_pulse_measurements,
    create_processed_dir,
)


from stdp_photonics.config import toml_config, cli_config
from stdp_photonics.devices import SoaSimulationParams, SoaSimulationResult, SoaSimulation
import stdp_photonics.signals.signals as signals
from stdp_photonics.plots import (
    plot_carrier_density, 
    plot_probe_gain_dB, 
    plot_input_output,
    plot_modal_gain_minus_alpha, 
    plot_energy_vs_delta_t,
    plot_2_pulses,
    plot_pulse_2_delta_t_sweep,
)

# Cases to be run (can be overriden through cli)
# Each case name corresponds to a toml table in a toml file with the follwing subtables:
# [soa_material]
# [soa_conditions]
# [soa_dims]
# [sim_params]

# For example, for cases=["case_a","case_b"] the toml structure should be:
# [case_a]
# [case_a.soa_conditions]
# ... (see SoaConditions dataclass for the values of this section)
# [case_a.soa_material]
# ... (see SoaMaterial dataclass for the values of this section)
# [case_a.soa_dims]
# ... (see SoaDimensions dataclass for the values of this section)
# [case_a.soa_sim_params]
# ... (see SinglePulseSoaGainSimulationParams dataclass for the values of this section)
CASES = [
    "case_a",
    "case_b", 
    "case_c", 
    "case_d",
]

@cli_config
@toml_config(section="soa_sim_params")
@dataclass
class TwoPulsesSweepSoaSimulationParams(SoaSimulationParams):
    sim_type_name : str = "2_pulses_sweep"

    # Pulse_1
    pulse_1_wavelength: float = 1551e-9
    pulse_1_power_peak: float = 5e-3
    pulse_1_width: float = 65e-12
    pulse_1_t_peak: float = 200e-12
    pulse_1_forward : bool = True

    # Pulse_2
    pulse_2_wavelength: float = 1560e-9
    pulse_2_power_peak: float = 5e-3
    pulse_2_width: float = 65e-12
    pulse_2_forward : bool = True
    pulse_2_t_peak_min : float = 100e-12
    pulse_2_t_peak_max : float = 400e-12
    pulse_2_t_peak_N : int = 4

    i : int = 0 # sweep iteration. This determines the time of the 2 pulse
                # current_iteration_pulse_2_t_peak = self.pulse_2_t_peak_sweep[self.i]
    
    @property
    def pulse_2_t_peak_sweep(self):
        return np.linspace(
            self.pulse_2_t_peak_min,
            self.pulse_2_t_peak_max,
            self.pulse_2_t_peak_N,
        )

    @property
    def pulse_2_t_peak(self):
        return self.pulse_2_t_peak_sweep[self.i]

    @property
    def input_signals_forward_list(self) -> list[Callable[[float], np.ndarray]]:
        # Input signals
        ## Pulse 1
        pulses_forward = []
        if self.pulse_1_forward:
            pulse_1_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
                signal_power=lambda t : signals.gaussian_pulse(
                    t,
                    power_peak=self.pulse_1_power_peak,
                    pulse_width=self.pulse_1_width,
                    t_peak=self.pulse_1_t_peak,
                ),
                spectral_position=self.pulse_1_wavelength
            )
            pulses_forward.append(pulse_1_signal)

        ## Pulse 2
        if self.pulse_2_forward:
            
            pulse_2_t_peak = self.pulse_2_t_peak
            pulse_2_power_peak = self.pulse_2_power_peak
            pulse_2_wavelength = self.pulse_2_wavelength
            pulse_2_width = self.pulse_2_width

            pulse_2_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
                signal_power=lambda t : signals.gaussian_pulse(
                    t,
                    power_peak=pulse_2_power_peak,
                    pulse_width=pulse_2_width,
                    t_peak=pulse_2_t_peak,
                ),
                spectral_position=pulse_2_wavelength
            )
            pulses_forward.append(pulse_2_signal)

        return pulses_forward

    @property
    def input_signals_backward_list(self) -> list[Callable[[float], np.ndarray]]:
        # Input signals
        ## Pulse 1
        pulses_backward = []
        if not self.pulse_1_forward:
            pulse_1_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
                signal_power=lambda t : signals.gaussian_pulse(
                    t,
                    power_peak=self.pulse_1_power_peak,
                    pulse_width=self.pulse_1_width,
                    t_peak=self.pulse_1_t_peak,
                ),
                spectral_position=self.pulse_1_wavelength
            )
            pulses_backward.append(pulse_1_signal)

        ## Pulse 2
        if not self.pulse_2_forward:
            pulse_2_t_peak = self.pulse_2_t_peak
            pulse_2_power_peak = self.pulse_2_power_peak
            pulse_2_wavelength = self.pulse_2_wavelength
            pulse_2_width = self.pulse_2_width
            pulse_2_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
                signal_power=lambda t : signals.gaussian_pulse(
                    t,
                    power_peak=pulse_2_power_peak,
                    pulse_width=pulse_2_width,
                    t_peak=pulse_2_t_peak,
                ),
                spectral_position=pulse_2_wavelength
            )
            pulses_backward.append(pulse_2_signal)

        return pulses_backward


# TwoPulsesSoaSimulationResults = namedtuple(
#     "TwoPulsesSoaSimulationResults",
#     """
#     sim_params t z spectrum
#     power_forward_by_wavelength power_backward_by_wavelength power_total_by_wavelength power_forward power_backward power_total 
#     input_power_forward_by_wavelength input_power_backward_by_wavelength output_power_forward_by_wavelength output_power_backward_by_wavelength 
#     input_power_forward input_power_backward output_power_forward output_power_backward 
#     input_power_pulse_1 output_power_pulse_1 input_power_pulse_2 output_power_pulse_2 
#     input_pulse_1_measurements output_pulse_1_measurements input_pulse_2_measurements output_pulse_2_measurements
#     """
# )

def simulate(exp_setup : ExperimentSetup) -> dict[str, dict[str, SoaSimulationResult]]:
    
    results : dict[str, dict[str, SoaSimulationResult]] = {}
    
    for case in exp_setup.cases:
        if VERBOSE:
            print(f"{case}: running simulation")

        results[case] = {}

        soa_conditions = exp_setup.soa_conditions[case]
        soa_material = exp_setup.soa_material[case]
        soa_dims = exp_setup.soa_dims[case]
        sim_params_base = exp_setup.sim_params[case]

        save_dir = exp_setup.outdir / exp_setup.full_run_name / case 
        save_configs(soa_conditions, soa_material, soa_dims, sim_params_base, save_dir)

        #########################################
        # Simulation to find initial condition  #
        #########################################

        sim_params_for_init_cond : TwoPulsesSweepSoaSimulationParams = copy.copy(sim_params_base)
        sim_params_for_init_cond.t_start = 0
        sim_params_for_init_cond.t_end = 3e-3
        sim_params_for_init_cond.pulse_1_power_peak = 0.0
        sim_params_for_init_cond.pulse_2_power_peak = 0.0

        sim_for_init_cond : SoaSimulation = SoaSimulation(
            sim_params=sim_params_for_init_cond,
            soa_conditions=soa_conditions,
            soa_material=soa_material,
            soa_dims=soa_dims,
        )

        ## Simulate SOA propagation with probe and without pump to find steady state n(z)
        result_sim_for_init_cond = sim_for_init_cond.run()

        ##################################################
        # Run main simulations                           #
        ##################################################

        for i in range(sim_params_base.pulse_2_t_peak_N):
            sim_params_main = copy.copy(sim_params_base)
            sim_params_main.i = i

            single_run_name = str(i)
            
            # Use steady state n(z) (without input signals) as initial condition for main simulation
            sim_params_main.n_initial = result_sim_for_init_cond.n[:, -1]

            # sim_params_main_sweep.append(sim_params_main)

            # Run main simulation
            sim_main = SoaSimulation(
                sim_params=sim_params_main,
                soa_conditions=soa_conditions,
                soa_material=soa_material,
                soa_dims=soa_dims,
            )
            sim_result_main = sim_main.run()

            results[case][single_run_name] = sim_result_main

            ##################################################
            # Save and return simulation results             #
            ##################################################

            save_dir = exp_setup.outdir / exp_setup.full_run_name / case / single_run_name
            save_dir.mkdir(parents=True, exist_ok=True)
            save_soa_sim_result(sim_result_main, save_dir=save_dir)

    return results

# TODO: get exp_setup from saved config files
def process(exp_setup : ExperimentSetup, results : dict[str, list[SoaSimulationResult]]):

    for case, case_results in results.items():
        if VERBOSE:
            print(f"{case}: processing")

        ##############################################################
        # Processing and saving each simulation results separately   #
        ##############################################################
        # ___results_sim_main_sweep : list[TwoPulsesSoaSimulationResults] = []
        # soa_conditions = exp_setup.soa_conditions[case]
        soa_material = exp_setup.soa_material[case]
        # soa_dims = exp_setup.soa_dims[case]
        sim_params_base = exp_setup.sim_params[case]
        spectrum : DiscreteSpectrum = sim_params_base.spectrum
        pulse_1_spectrum_idx = spectrum.get_index(sim_params_base.pulse_1_wavelength)
        pulse_2_spectrum_idx = spectrum.get_index(sim_params_base.pulse_2_wavelength)

        case_pulse_measurements : dict[str, dict[str, dict[str, float]]] = {}    # case_pulse_measurements[single_run_name][pulse_name][measurement]
        case_pulse_powers : dict[str, dict[str,np.ndarray]] = {}                 # case_pulse_powers[single_run_name][pulse_signal_name]
        case_t : dict[str,np.ndarray] = {}                                       # case_t[single_run_name]

        case_save_dir = exp_setup.outdir / exp_setup.full_run_name / case

        for i, (single_run_name, result) in enumerate(case_results.items()):
            sim_params = copy.copy(sim_params_base)
            sim_params.i = i
            save_dir = create_processed_dir(case_save_dir / single_run_name)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            ###
            # Get SOA power (forward and backward) for all t, z, and wavelength:
            ###
            
            n = result.n
            t_eval = result.t
            z = result.z
            power_forward_by_wavelength = result.P_forward      #[t, z, wavelength]
            power_backward_by_wavelength = result.P_backward    #[t, z, wavelength]

            case_t[single_run_name] = t_eval
            ###
            # Calc. and save pulse measurements
            ###

            ## Calc. measurements
            if sim_params.pulse_1_forward:
                input_power_pulse_1 = power_forward_by_wavelength[:,0,pulse_1_spectrum_idx]
                output_power_pulse_1 = power_forward_by_wavelength[:,-1,pulse_1_spectrum_idx]
            else:
                input_power_pulse_1 = power_backward_by_wavelength[:,-1,pulse_1_spectrum_idx]
                output_power_pulse_1 = power_backward_by_wavelength[:,0,pulse_1_spectrum_idx]

            if sim_params.pulse_2_forward:
                input_power_pulse_2 = power_forward_by_wavelength[:,0,pulse_2_spectrum_idx]
                output_power_pulse_2 = power_forward_by_wavelength[:,-1,pulse_2_spectrum_idx]
            else:
                input_power_pulse_2 = power_backward_by_wavelength[:,-1,pulse_2_spectrum_idx]
                output_power_pulse_2 = power_backward_by_wavelength[:,0,pulse_2_spectrum_idx]

            case_pulse_powers[single_run_name] = {}
            case_pulse_powers[single_run_name]["input_power_pulse_1"] = input_power_pulse_1
            case_pulse_powers[single_run_name]["output_power_pulse_1"] = output_power_pulse_1
            case_pulse_powers[single_run_name]["input_power_pulse_2"] = input_power_pulse_2
            case_pulse_powers[single_run_name]["output_power_pulse_2"] = output_power_pulse_2

            input_pulse_1_measurements = signals.measure_pulse(input_power_pulse_1, t_eval)
            output_pulse_1_measurements = signals.measure_pulse(output_power_pulse_1, t_eval)
            input_pulse_2_measurements = signals.measure_pulse(input_power_pulse_2, t_eval)
            output_pulse_2_measurements = signals.measure_pulse(output_power_pulse_2, t_eval)
            
            ## Save measurements
            single_run_pulse_measurements = {}
            single_run_pulse_measurements["input_pulse_1_measurements"] = input_pulse_1_measurements
            single_run_pulse_measurements["output_pulse_1_measurements"] = output_pulse_1_measurements
            single_run_pulse_measurements["input_pulse_2_measurements"] = input_pulse_2_measurements
            single_run_pulse_measurements["output_pulse_2_measurements"] = output_pulse_2_measurements

            case_pulse_measurements[single_run_name] = single_run_pulse_measurements

            save_pulse_measurements(
                pulse_measurements=single_run_pulse_measurements,
                save_dir=save_dir,
                filename="P_pulses_input_output",
            )

            ###
            # Calc. and save SOA input and output (forward, backward) for all t:
            ###

            input_power_forward_by_wavelength = power_forward_by_wavelength[:,0,:]      #[t, wavelength]
            output_power_forward_by_wavelength = power_forward_by_wavelength[:,-1,:]    #[t, wavelength]
            input_power_backward_by_wavelength = power_backward_by_wavelength[:,-1,:]   #[t, wavelength]
            output_power_backward_by_wavelength = power_backward_by_wavelength[:,0,:]   #[t, wavelength]

            input_power_forward = np.sum(input_power_forward_by_wavelength, axis=1)     #[t]
            output_power_forward = np.sum(output_power_forward_by_wavelength, axis=1)   #[t]
            input_power_backward = np.sum(input_power_backward_by_wavelength, axis=1)   #[t]
            output_power_backward = np.sum(output_power_backward_by_wavelength, axis=1) #[t]

            # Plot and save input and output power signals
            plot_input_output(
                input=input_power_forward,
                output=output_power_forward,
                t=t_eval,
                title="Input and output (forward propagation)",
                outfile=save_dir / "power_forward_propagation.svg",
            )
            plot_input_output(
                input=input_power_backward,
                output=output_power_backward,
                t=t_eval,
                title="Input and output (backward propagation)",
                outfile = save_dir / "power_backward_propagation.svg",
            )
            # Plot and save input and output power signals (normalized)
            input_power_forward_max = np.max(input_power_forward)
            input_power_forward_normalized = input_power_forward/np.max(input_power_forward) if input_power_forward_max > 1e-18 else np.zeros_like(input_power_forward)
            input_power_backward_max = np.max(input_power_backward)
            input_power_backward_normalized = input_power_backward/np.max(input_power_backward) if input_power_backward_max > 1e-18 else np.zeros_like(input_power_backward)

            plot_input_output(
                input=input_power_forward_normalized,
                output=output_power_forward/np.max(output_power_forward),
                t=t_eval,
                ylabel="Normalized power [a.u.]",
                title="Normalized input and output (forward propagation)",
                outfile=save_dir / "power_forward_propagation_normalized.svg",
            )
            plot_input_output(
                input=input_power_backward_normalized,
                output=output_power_backward/np.max(output_power_backward),
                t=t_eval,
                ylabel="Normalized power [a.u.]",
                title="Normalized input and output (backward propagation)",
                outfile = save_dir / "power_backward_propagation_normalized.svg",
            )

            ###
            # Modal gain and loss
            ###

            plot_modal_gain_minus_alpha(
                n_min=np.min(n),
                n_max=np.max(n),
                wavelength_min=sim_params.spectrum_wavelength_min,
                wavelength_max=sim_params.spectrum_wavelength_max,
                modal_gain=soa_material.modal_gain,
                alpha = soa_material.alpha,
                title="gm(n,λ)-α",
                outfile=save_dir / "modal_gain_minus_alpha.svg",
            )

            plot_modal_gain_minus_alpha(
                n_min=np.min(n),
                n_max=np.max(n),
                wavelength_min=spectrum.min,
                wavelength_max=spectrum.max,
                modal_gain=soa_material.modal_gain,
                alpha = 0,
                title="gm(n,λ)",
                outfile=save_dir / "modal_gain.svg",
            )

            ###
            # Carrier density
            ###
            plot_carrier_density(
                t_eval,
                z,
                n,
                outfile=save_dir / "carrier_density.png",
            )

            plot_2_pulses(
                input_power_pulse_1=input_power_pulse_1,
                input_power_pulse_2=input_power_pulse_2,
                output_power_pulse_1=output_power_pulse_1,
                output_power_pulse_2=output_power_pulse_2,
                t=t_eval,
            )

            # TODO: maybe add backward propagating probe?

            # ### Probe gain dB ###
            # plot_probe_gain_dB(
            #     t_eval, 
            #     probe_gain_dB=backward_gain_dB,
            #     pump_output=output_power_forward,
            #     show=False,
            #     outfile=save_dir / "probe_gain_dB.svg",
            # )


        ##################################################
        # Processing and saving sweep results combined   #
        ##################################################
        save_dir = create_processed_dir(case_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        energies_pulse_2 = [v["output_pulse_2_measurements"]["energy"] for v in case_pulse_measurements.values()]
        delta_t_peaks = sim_params_base.pulse_2_t_peak_sweep - sim_params_base.pulse_1_t_peak

        plot_energy_vs_delta_t(
            energy=energies_pulse_2, 
            delta_t=delta_t_peaks,
            outfile=save_dir / "energy_vs_delta_t.svg"
        )

        ts_minus_t1 = [t - sim_params_base.pulse_1_t_peak for t in case_t.values()]
        output_powers_pulse_2 = [v["output_power_pulse_2"] for v in case_pulse_powers.values()]
        plot_pulse_2_delta_t_sweep(
            ts_minus_t1, 
            output_powers_pulse_2, 
            delta_t_peaks,
            figsize=(6.0, 4.0),
            outfile=save_dir / "2_pulses_sweep.svg"
        )


def main():

    parser = ArgumentParser()
    add_args_to_parser(parser,TwoPulsesSweepSoaSimulationParams)
    args = parser.parse_args()

    exp_setup = ExperimentSetup.from_args(args, TwoPulsesSweepSoaSimulationParams, CASES)

    if exp_setup.sim_data_dir:
        results = load_simulation_results_from_dir(exp_setup.sim_data_dir)
        exp_setup.update_from_dir(TwoPulsesSweepSoaSimulationParams)
    else:
        results = simulate(exp_setup)

    if exp_setup.process:
        process(exp_setup=exp_setup, results=results)

if __name__ == '__main__':
    main()