DEBUG = True
OUTDIR = "./out"

import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass, field
from collections.abc import Callable
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
from experiment_setup import ExperimentSetup, add_args_to_parser
from results_manager import (
    save_n, 
    save_configs, 
    save_power, 
    save_power_spectral, 
    save_input_output_power, 
    save_input_output_power_spectral, 
    save_t_and_z,
    save_pulse_measurements,
)
if DEBUG:
    import time

from config import toml_config
from config import cli_config
import copy
from devices import SoaSimulationParams, SoaMaterial, SoaConditions, SoaSimulation, SoaDimensions
import signals.signals as signals
from plots import plot_carrier_density, plot_probe_gain_dB, plot_signal_output, plot_modal_gain_minus_alpha, lighten_color

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
    pulse_2_t_peak_N : float = 4

    pulse_2_t_peak_sweep : np.ndarray = field(init=False)

    i : int = 0 # sweep iteration. This determines the time of the 2 pulse
                # current_iteration_pulse_2_t_peak = self.pulse_2_t_peak_sweep[self.i]
    
    def __post_init__(self):
        self.pulse_2_t_peak_sweep = np.linspace(self.pulse_2_t_peak_min, self.pulse_2_t_peak_max, self.pulse_2_t_peak_N)

    @property
    def pulse_2_t_peak(self):
        return self.pulse_2_t_peak_sweep[self.i]

    @property
    def input_signals_forward_list(self):
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
        i = self.i
        if self.pulse_2_forward:
            pulse_2_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
                signal_power=lambda t : signals.gaussian_pulse(
                    t,
                    power_peak=self.pulse_2_power_peak,
                    pulse_width=self.pulse_2_width,
                    t_peak=self.pulse_2_t_peak_sweep[i],
                ),
                spectral_position=self.pulse_2_wavelength
            )
            pulses_forward.append(pulse_2_signal)

        return pulses_forward

    @property
    def input_signals_backward_list(self):
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
        i = self.i
        if not self.pulse_2_forward:
            pulse_2_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
                signal_power=lambda t : signals.gaussian_pulse(
                    t,
                    power_peak=self.pulse_2_power_peak,
                    pulse_width=self.pulse_2_width,
                    t_peak=self.pulse_2_t_peak_sweep[i],
                ),
                spectral_position=self.pulse_2_wavelength
            )
            pulses_backward.append(pulse_2_signal)

        return pulses_backward

TwoPulsesSoaSimulationResults = namedtuple(
    "TwoPulsesSoaSimulationResults",
    """
    sim_params t z spectrum
    power_forward_by_wavelength power_backward_by_wavelength power_total_by_wavelength power_forward power_backward power_total 
    input_power_forward_by_wavelength input_power_backward_by_wavelength output_power_forward_by_wavelength output_power_backward_by_wavelength 
    input_power_forward input_power_backward output_power_forward output_power_backward 
    input_power_pulse_1 output_power_pulse_1 input_power_pulse_2 output_power_pulse_2 
    input_pulse_1_measurements output_pulse_1_measurements input_pulse_2_measurements output_pulse_2_measurements
    """
)

def simulate():
    pass

def main():

    parser = ArgumentParser()
    add_args_to_parser(parser,TwoPulsesSweepSoaSimulationParams)
    args = parser.parse_args()

    exp_setup = ExperimentSetup.from_args(args, TwoPulsesSweepSoaSimulationParams, CASES)

    for case in exp_setup.cases:
        if DEBUG:
            print(case)

        soa_conditions = exp_setup.soa_conditions[case]
        soa_material = exp_setup.soa_material[case]
        soa_dims = exp_setup.soa_dims[case]
        sim_params_base = exp_setup.sim_params[case]

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
        solution_sim_for_init_cond = sim_for_init_cond.run()

        #########################################
        # Run main simulations                  #
        #########################################

        sim_params_main_sweep : list[TwoPulsesSweepSoaSimulationParams] = []

        for i in range(sim_params_base.pulse_2_t_peak_N):
            sim_params_main = copy.copy(sim_params_base)
            sim_params_main.i = i
            
            # Use steady state n(z) (without input signals) as initial condition for main simulation
            sim_params_main.n_initial = solution_sim_for_init_cond.y[:, -1]

            # Add to param sweep list
            sim_params_main_sweep.append(sim_params_main)

        sim_main_sweep = [
            SoaSimulation(
                sim_params=sim_params_main,
                soa_conditions=soa_conditions,
                soa_material=soa_material,
                soa_dims=soa_dims,
            )
            for sim_params_main in sim_params_main_sweep
        ]

        # Run main simulation
        solution_sim_main_sweep = [sim_main.run() for sim_main in sim_main_sweep]


        ##############################################################
        # Processing and saving each simulation results separately   #
        ##############################################################

        results_sim_main_sweep : list[TwoPulsesSoaSimulationResults] = []

        for sim_params_main, sim_main, solution_sim_main  in zip(sim_params_main_sweep, sim_main_sweep, solution_sim_main_sweep):
            
            ###
            # Get SOA power (forward, backward, and total) for all t and z:
            ###
            
            n = solution_sim_main.y
            t_eval = solution_sim_main.t
            z = sim_main.z

            power_forward_by_wavelength, power_backward_by_wavelength, power_total_by_wavelength = sim_main.get_powers_from_solution(n, t_eval)

            power_forward = np.sum(power_forward_by_wavelength, axis=2)
            power_backward = np.sum(power_backward_by_wavelength, axis=2)
            power_total = power_forward + power_backward

            ###
            # Get SOA input and output (forward, backward) for all t:
            ###

            output_power_forward_by_wavelength = power_forward_by_wavelength[:,-1,:]    #[t, wavelength]
            output_power_backward_by_wavelength = power_backward_by_wavelength[:,0,:]   #[t, wavelength]

            output_power_forward = np.sum(output_power_forward_by_wavelength, axis=1)   #[t]
            output_power_backward = np.sum(output_power_backward_by_wavelength, axis=1) #[t]

            ## Forward input power
            input_power_forward_by_wavelength = np.array([sim_main.input_signal_forward(t) for t in t_eval])
            input_power_forward = np.array(np.sum(input_power_forward_by_wavelength, axis=1)) # sum across spectrum to get total power

            ## Backwards input power
            input_power_backward_by_wavelength = np.array([sim_main.input_signal_backward(t) for t in t_eval])
            input_power_backward = np.array(np.sum(input_power_backward_by_wavelength, axis=1)) # sum across spectrum to get total power

            ###
            # Get pulses measurements:
            ###

            ## Filter input and output power in each pulses wavelength
            spectrum_index_pulse_1 = sim_main.spectrum.get_index(sim_params_main.pulse_1_wavelength)
            spectrum_index_pulse_2 = sim_main.spectrum.get_index(sim_params_main.pulse_2_wavelength)

            if sim_params_main.pulse_1_forward:
                output_power_pulse_1 = output_power_forward_by_wavelength[:,spectrum_index_pulse_1]
                input_power_pulse_1 = input_power_forward_by_wavelength[:,spectrum_index_pulse_1]
            else:
                output_power_pulse_1 = output_power_backward_by_wavelength[:,spectrum_index_pulse_1]
                input_power_pulse_1 = input_power_backward_by_wavelength[:,spectrum_index_pulse_1]
            
            if sim_params_main.pulse_2_forward:
                output_power_pulse_2 = output_power_forward_by_wavelength[:,spectrum_index_pulse_2]
                input_power_pulse_2 = input_power_forward_by_wavelength[:,spectrum_index_pulse_2]
            else:
                output_power_pulse_2 = output_power_backward_by_wavelength[:,spectrum_index_pulse_2]
                input_power_pulse_2 = input_power_backward_by_wavelength[:,spectrum_index_pulse_2]


            ## Calc. measurements
            input_pulse_1_measurements = signals.measure_pulse(input_power_pulse_1, t_eval)
            output_pulse_1_measurements = signals.measure_pulse(output_power_pulse_1, t_eval)
            input_pulse_2_measurements = signals.measure_pulse(input_power_pulse_2, t_eval)
            output_pulse_2_measurements = signals.measure_pulse(output_power_pulse_2, t_eval)

            ###
            # Add this iteration result to the list of all results
            ###

            results_sim_main_sweep.append(
                TwoPulsesSoaSimulationResults(
                    sim_params_main,t_eval,z,sim_main.spectrum,
                    power_forward_by_wavelength,power_backward_by_wavelength,power_total_by_wavelength,power_forward,power_backward,power_total,
                    input_power_forward_by_wavelength,input_power_backward_by_wavelength,output_power_forward_by_wavelength,output_power_backward_by_wavelength,
                    input_power_forward,input_power_backward,output_power_forward,output_power_backward,
                    input_power_pulse_1,output_power_pulse_1,input_power_pulse_2,output_power_pulse_2,
                    input_pulse_1_measurements,output_pulse_1_measurements,input_pulse_2_measurements,output_pulse_2_measurements,
                )
            )

            ###
            # Save results
            ###

            save_dir = Path(OUTDIR) / exp_setup.full_run_name / case / str(sim_params_main.i)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configs
            save_configs(soa_conditions, soa_material, soa_dims, sim_params_base, save_dir)
            
            # Save t and z
            save_t_and_z(t=t_eval, z=z, save_dir=save_dir)

            # Save n
            save_n(n, z, t_eval, save_dir)
            
            # Save powers for all t and z, discriminated by wavelength
            save_power_spectral(
                signals_P=[ power_forward_by_wavelength, power_backward_by_wavelength, power_total_by_wavelength],
                signals_names=["P_forward", "P_backward", "P"],
                t=t_eval,
                z=z,
                spectrum=sim_main.spectrum,
                save_dir=save_dir,
                filename="P_by_wavelength",
            )

            # Save powers adding across whole spectrum
            save_power(
                signals_P=[power_forward, power_backward, power_total],
                signals_names=["P_forward", "P_backward", "P"],
                t=t_eval,
                z=z,
                save_dir=save_dir,
                filename="P"
            )

            # Save inputs and outputs for all wavelengths
            save_input_output_power(
                signals_P=(
                    (input_power_forward, output_power_forward),
                    (input_power_backward, output_power_backward),
                ),
                signals_names=("P_forward", "P_backward"),
                t=t_eval,
                save_dir=save_dir,
                filename="P_input_output",
            )

            # Save inputs and outputs only for pulses wavlengths
            save_input_output_power(
                signals_P=(
                    (input_power_pulse_1, output_power_pulse_1),
                    (input_power_pulse_2, output_power_pulse_2),
                ),
                signals_names=("P_pulse_1", "P_pulse_2"),
                t=t_eval,
                save_dir=save_dir,
                filename="P_pulses_input_output",
            )

            # Save pulse measurements
            save_pulse_measurements(
                pulse_measurements=(
                    input_pulse_1_measurements,
                    output_pulse_1_measurements,
                    input_pulse_2_measurements,
                    output_pulse_2_measurements,
                ),
                signals_names=( "pulse_1_input", "pulse_1_output", "pulse_2_input", "pulse_2_output" ),
                save_dir=save_dir,
                filename="P_pulses_input_output",
            )

            ###
            # Create and save plots
            ###

            # Modal gain - losses
            plot_modal_gain_minus_alpha(
                n_min=np.min(n),
                n_max=np.max(n),
                wavelength_min=sim_main.spectrum.min,
                wavelength_max=sim_main.spectrum.max,
                modal_gain=soa_material.modal_gain,
                alpha = soa_material.alpha,
                title="gm(n,λ)-α",
                outfile=save_dir / "modal_gain_minus_alpha.svg",
            )

            # Only modal gain
            plot_modal_gain_minus_alpha(
                n_min=np.min(n),
                n_max=np.max(n),
                wavelength_min=sim_main.spectrum.min,
                wavelength_max=sim_main.spectrum.max,
                modal_gain=soa_material.modal_gain,
                alpha = 0,
                title="gm(n,λ)",
                outfile=save_dir / "modal_gain_minus_alpha.svg",
            )

            # Carrier density
            plot_carrier_density(
                t_eval,
                z,
                n,
                outfile=save_dir / "carrier_density.png",
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

            # Forward input and output
            _, ax = plot_signal_output(
                t_eval,
                input_power_forward,
                label="Input",
                # outfile=save_dir / "forward_input_output.svg",
                show=False,
            )

            plot_signal_output(
                t_eval,
                output_power_forward,
                ax=ax,
                label="Output",
                title="Forward signal",
                outfile=save_dir / "forward_input_output_total_power.svg",
                show=False,
            )

            # Forward input and output (normalized)
            _, ax = plot_signal_output(
                t_eval,
                input_power_forward/max(input_power_forward),
                label="Input",
                title="Forward signal",
                # outfile=save_dir / "forward_input_output_normalized.svg",
                show=False,
            )

            plot_signal_output(
                t_eval,
                output_power_forward/max(output_power_forward),
                ax=ax,
                label="Output",
                ylabel="Normalized power [a.u.]",
                outfile=save_dir / "forward_input_output_total_power_normalized.svg",
                show=False,
            )

            # Forward input and output filtered at pulses wavelengths
            pulse_1_output_color = 'b'
            pulse_1_input_color = lighten_color(pulse_1_output_color)
            pulse_2_output_color = 'r'
            pulse_2_input_color = lighten_color(pulse_2_output_color)
            input_linestyle = '-'
            output_linestyle = '--'
            _, ax = plot_signal_output(
                t_eval,
                input_power_pulse_1,
                label="Input pulse 1",
                # outfile=save_dir / "forward_input_output.svg",
                linestyle=input_linestyle,
                color=pulse_1_input_color,
                show=False,
            )

            plot_signal_output(
                t_eval,
                output_power_pulse_1,
                ax=ax,
                label="Output pulse 1",
                # outfile=save_dir / "forward_input_output.svg",
                color=pulse_1_output_color,
                linestyle=output_linestyle,
                show=False,
            )

            plot_signal_output(
                t_eval,
                input_power_pulse_2,
                ax=ax,
                label="Input pulse 2",
                # outfile=save_dir / "forward_input_output.svg",
                linestyle=input_linestyle,
                color=pulse_2_input_color,
                show=False,
            )

            plot_signal_output(
                t_eval,
                output_power_pulse_2,
                ax=ax,
                label="Output pulse 2",
                title="2 pulses input and output powers",
                outfile=save_dir / "forward_input_output.svg",
                color=pulse_2_output_color,
                linestyle=output_linestyle,
                show=False,
            )


        ##################################################
        # Processing and saving sweep results combined   #
        ##################################################


        save_dir = Path(OUTDIR) / exp_setup.full_run_name / case
        save_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Sweep energy vs delta t
        # TODO: All pulses 2 powers vs. t + vertial line on t_peak_1 and all t_peak_2

        ts = [result.t for result in results_sim_main_sweep]
        input_powers_pulse_2 = [result.input_power_pulse_2 for result in results_sim_main_sweep]
        output_powers_pulse_2 = [result.output_power_pulse_2 for result in results_sim_main_sweep]
        t_peaks_pulse_2 = [result.sim_params.pulse_2_t_peak for result in results_sim_main_sweep]
        deltas_t_peaks = [result.sim_params.pulse_2_t_peak - result.sim_params.pulse_1_t_peak for result in results_sim_main_sweep]
        energies_pulse_2 = [result.output_pulse_2_measurements["energy"] for result in results_sim_main_sweep]

        # Forward input and output filtered at pulses wavelengths
        pulse_1_output_color = 'k'
        pulse_1_input_color = lighten_color(pulse_1_output_color)
        input_linestyle = '-'
        output_linestyle = '--'

        t = results_sim_main_sweep[-1].t
        input_power_pulse_1 = results_sim_main_sweep[-1].input_power_pulse_1
        output_power_pulse_1 = results_sim_main_sweep[-1].output_power_pulse_1
        
        fig, ax = plot_signal_output(
            t,
            input_power_pulse_1,
            label="Input pulse 1",
            linestyle=input_linestyle,
            color=pulse_1_input_color,
            show=False,
        )

        plot_signal_output(
            t,
            output_power_pulse_1,
            ax=ax,
            label="Output pulse 1",
            # outfile=save_dir / "forward_input_output.svg",
            color=pulse_1_output_color,
            linestyle=output_linestyle,
            show=False,
        )

        for i, (
                t, 
                output_power_pulse_2, 
                input_power_pulse_2, 
                t_peak_pulse_2, 
                delta_t_peak
            ) in enumerate(zip(
                ts,
                output_powers_pulse_2, 
                input_powers_pulse_2, 
                t_peaks_pulse_2, 
                deltas_t_peaks
            )
        ):
            color_idx = i % 10
            pulse_2_output_color = f'C{color_idx}'
            pulse_2_input_color = lighten_color(pulse_2_output_color)

            plot_signal_output(
                t,
                input_power_pulse_2,
                label=f"Pulse 2 input (Δt_peak={delta_t_peak*1e12:.0f} ps)",
                linestyle=input_linestyle,
                color=pulse_2_input_color,
                ax=ax,
                show=False,
            )

            plot_signal_output(
                t,
                output_power_pulse_2,
                ax=ax,
                label=f"Pulse 2 output (Δt_peak={delta_t_peak*1e12:.0f} ps)",
                # outfile=save_dir / "forward_input_output.svg",
                linestyle=output_linestyle,
                color=pulse_2_output_color,
                show=False,
            )

            ax.axvline(x=t_peak_pulse_2*1e12, color=pulse_2_output_color)

        fig.savefig(save_dir/f"sweep_output_powers.svg")


if __name__ == '__main__':
    main()