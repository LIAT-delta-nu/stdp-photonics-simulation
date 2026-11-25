DEBUG = True
OUTDIR = "./out"

import numpy as np
import matplotlib.pyplot as plt
import csv
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
# import devices.SOA_model_Li_2008 as soa
from devices import SoaSimulationParams, SoaMaterial, SoaConditions, SoaSimulation, SoaDimensions
from devices.SOA_model_Li_2008 import soa_propagate_power
import signals.signals as signals
from spectrum.spectrum import DiscreteSpectrum
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
class TwoPulsesSoaSimulationParams(SoaSimulationParams):
    sim_type_name : str = "2 pulses"

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
    pulse_2_t_peak: float = 300e-12
    pulse_2_forward : bool = True

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
        if self.pulse_2_forward:
            pulse_2_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
                signal_power=lambda t : signals.gaussian_pulse(
                    t,
                    power_peak=self.pulse_2_power_peak,
                    pulse_width=self.pulse_2_width,
                    t_peak=self.pulse_2_t_peak,
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
        if not self.pulse_2_forward:
            pulse_2_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
                signal_power=lambda t : signals.gaussian_pulse(
                    t,
                    power_peak=self.pulse_2_power_peak,
                    pulse_width=self.pulse_2_width,
                    t_peak=self.pulse_2_t_peak,
                ),
                spectral_position=self.pulse_2_wavelength
            )
            pulses_backward.append(pulse_2_signal)

        return pulses_backward


def build_parser():
    parser = ArgumentParser()
    TwoPulsesSoaSimulationParams.add_cli_args_to_parser(parser)

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--default-case",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument("--config-file", type=Path, default="./sim_configs/SOA_simulation_Li_2008.toml")
    parser.add_argument("--soa-conditions-file", type=Path, default=None)
    parser.add_argument("--soa-material-file", type=Path, default=None)
    parser.add_argument("--soa-dims-file", type=Path, default=None)
    parser.add_argument("--sim-params-file", type=Path, default=None)

    return parser

def main():

    parser = build_parser()
    args = parser.parse_args()

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = input("Enter run name and press enter (if no run name is wanted just press enter):\n")

    cases = args.cases if args.cases else CASES
    default_case = args.default_case if args.default_case else cases[0]

    soa_conditions_file = args.soa_conditions_file if args.soa_conditions_file else args.config_file
    soa_material_file = args.soa_material_file if args.soa_material_file else args.config_file
    soa_dims_file = args.soa_dims_file if args.soa_dims_file else args.config_file
    sim_params_file = args.sim_params_file if args.sim_params_file else args.config_file

    # TODO: add cli args for SoaConditions, SoaMaterial and SoaDimensions
    default_soa_conditions=SoaConditions.from_toml(soa_conditions_file,f"{default_case}.soa_conditions")
    
    default_soa_material=SoaMaterial.from_toml(soa_material_file,f"{default_case}.soa_material")
    
    default_soa_dims=SoaDimensions.from_toml(soa_dims_file,f"{default_case}.soa_dimensions")
    
    default_sim_params : TwoPulsesSoaSimulationParams = TwoPulsesSoaSimulationParams.from_toml(sim_params_file,f"{default_case}.soa_sim_params")
    default_sim_params.update_from_cli_args(args)

    sim_type_name = default_sim_params.sim_type_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_run_name = f"{timestamp}__{sim_type_name}__{run_name}" if run_name else f"{timestamp}__{sim_type_name}"

    cases = list(dict.fromkeys(cases))  # Remove duplicates, preserving order

    for case in cases:
        if DEBUG:
            print(case)

        # First, simulate SOA propagation of only the probe to find the initial condition of the main simulation
        # Second, do the main simulation (2 pulses) using use the steady state of the probe-only simulation 
        # as initial n(z) 

        # TODO: add probe

        soa_conditions = copy.copy(default_soa_conditions)
        soa_material = copy.copy(default_soa_material)
        soa_dims = copy.copy(default_soa_dims)
        sim_params : TwoPulsesSoaSimulationParams = copy.copy(default_sim_params)

        soa_conditions.update_from_toml(soa_conditions_file,f"{case}.soa_conditions")
        soa_material.update_from_toml(soa_material_file,f"{case}.soa_material")
        soa_dims.update_from_toml(soa_dims_file,f"{case}.soa_dimensions")
        sim_params.update_from_toml(sim_params_file,f"{case}.soa_sim_params")
        sim_params.update_from_cli_args(args)

        #########################################
        # Simulation to find initial condition  #
        #########################################

        sim_params_initial_condition = copy.copy(sim_params)
        sim_params_initial_condition.t_start = 0
        sim_params_initial_condition.t_end = 3e-3
        sim_params_initial_condition.pulse_1_power_peak = 0.0
        sim_params_initial_condition.pulse_2_power_peak = 0.0

        sim_initial_condition = SoaSimulation(
            soa_conditions=soa_conditions,
            soa_material=soa_material,
            soa_dims=soa_dims,
            sim_params=sim_params_initial_condition,
        )

        ## Simulate SOA propagation with probe and without inputs to find steady state n(z)
        solution_sim_initial_condition = sim_initial_condition.run()

        #########################################
        # Main simulation                       #
        #########################################

        sim_params_main = copy.copy(sim_params)
        
        # Use steady state n(z) (without input signals) as initial condition for main simulation
        sim_params_main.n_initial = solution_sim_initial_condition.y[:, -1]

        sim_main = SoaSimulation(
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

        # Calc. output
        n = solution_sim_main.y
        t_eval = solution_sim_main.t

        if DEBUG:
            start = time.perf_counter()

        input_signal_backward = sim_main.input_signal_backward
        input_signal_forward = sim_main.input_signal_forward
        z = sim_main.z

        output_power_backward_by_wavelength = np.array([
            soa_propagate_power(
                input_signal_backward(t), 
                sim_main.spectrum,
                n,
                z,
                soa_material,
                soa_dims,
                forward=False,
            )[0]    # backwards propagated so output is at [0]
            for n, t in zip(n.T, t_eval)
        ])   #[t, wavelength]

        output_power_backward = np.sum(output_power_backward_by_wavelength, axis=1)

        if DEBUG:
            stop = time.perf_counter()
            elapsed = stop - start
            print(f"Backward propagation execution time: {elapsed} seconds.")
            start = time.perf_counter()

        output_power_forward_by_wavelength = np.array([
            soa_propagate_power(
                input_signal_forward(t),
                sim_main.spectrum,
                n,
                z,
                soa_material,
                soa_dims,
                forward=True,
            )[-1]   # forwards propagated so output is at [-1]
            for n, t in zip(n.T, t_eval)
        ])   #[t, wavelength]

        output_power_forward = np.sum(
            output_power_forward_by_wavelength,
            axis=1
        )

        if DEBUG:
            stop = time.perf_counter()
            elapsed = stop - start
            print(f"Forward propagation execution time: {elapsed} seconds.")

        ## Forward input power
        input_power_forward_by_wavelength = np.array([input_signal_forward(t) for t in t_eval])
        input_power_forward = np.array(np.sum(input_power_forward_by_wavelength, axis=1)) # sum across spectrum to get total power

        ## Backwards input power
        input_power_backward_by_wavelength = np.array([input_signal_backward(t) for t in t_eval])
        input_power_backward = np.array(np.sum(input_power_backward_by_wavelength, axis=1)) # sum across spectrum to get total power

        ## Filter input and output power in each pulses wavelength
        spectrum_index_pulse_1 = sim_main.spectrum.get_index(sim_params.pulse_1_wavelength)
        spectrum_index_pulse_2 = sim_main.spectrum.get_index(sim_params.pulse_2_wavelength)

        if sim_params.pulse_1_forward:
            output_power_pulse_1 = output_power_forward_by_wavelength[:,spectrum_index_pulse_1]
            input_power_pulse_1 = input_power_forward_by_wavelength[:,spectrum_index_pulse_1]
        else:
            output_power_pulse_1 = output_power_backward_by_wavelength[:,spectrum_index_pulse_1]
            input_power_pulse_1 = input_power_backward_by_wavelength[:,spectrum_index_pulse_1]
        
        if sim_params.pulse_2_forward:
            output_power_pulse_2 = output_power_forward_by_wavelength[:,spectrum_index_pulse_2]
            input_power_pulse_2 = input_power_forward_by_wavelength[:,spectrum_index_pulse_2]
        else:
            output_power_pulse_2 = output_power_backward_by_wavelength[:,spectrum_index_pulse_2]
            input_power_pulse_2 = input_power_backward_by_wavelength[:,spectrum_index_pulse_2]


        # ============================================================
        # Save configs
        # ============================================================
        save_dir = Path(OUTDIR) / full_run_name / str(case)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_outfile = save_dir / "config.toml"
        soa_conditions.to_toml(config_outfile)
        soa_material.to_toml(config_outfile)
        soa_dims.to_toml(config_outfile)
        sim_params.to_toml(config_outfile)

        # ============================================================
        # Save num results
        # ============================================================

        np.savez(
            save_dir / "n_sim_result.npz",
            t=t_eval,
            z=z,
            n=n,
        )
        np.savez(
            save_dir / "input_and_output_powers.npz",
                t=t_eval,
                input_power_forward=input_power_forward,
                output_power_forward=output_power_forward,
                input_power_backward=input_power_backward,
                output_power_backward=output_power_backward,
                input_power_pulse_1=input_power_pulse_1,
                output_power_pulse_1=output_power_pulse_1,
                input_power_pulse_2=input_power_pulse_2,
                output_power_pulse_2=output_power_pulse_2,
        )
        np.savetxt(
            save_dir / "n_sim_result.csv", 
            n,
            delimiter=',',
        )
        np.savetxt(
            save_dir / "n_sim_result_t.csv", 
            t_eval,
            delimiter=',',
        )
        np.savetxt(
            save_dir / "n_sim_result_z.csv", 
            z,
            delimiter=',',
        )
        data = np.block([
                arr[:, np.newaxis]
                for arr in (
                    t_eval,
                    input_power_forward,
                    output_power_forward,
                    input_power_backward,
                    output_power_backward,
                    input_power_pulse_1,
                    output_power_pulse_1,
                    input_power_pulse_2,
                    output_power_pulse_2,
                )
            ]
        )
        np.savetxt(
            save_dir / "pulse_1_and_2_powers.csv",
            data,
            comments='',
            delimiter=',',
            header="t[s], Input power forward [W], Output power forward [W], Input power backward [W], Output power backward [W], Input power pulse 1 [W], Output power pulse 1 [W], Input power pulse 2 [W], Output power pulse 2 [W],"
        )

        pulse_1_input_measurements = {"name": "pulse_1_input"}
        pulse_1_output_measurements = {"name":"pulse_1_output"}
        pulse_2_input_measurements = {"name": "pulse_2_input"}
        pulse_2_output_measurements = {"name":"pulse_2_output"}
        pulse_1_input_measurements |= signals.measure_pulse(input_power_pulse_1, t_eval)
        pulse_1_output_measurements |= signals.measure_pulse(output_power_pulse_1, t_eval)
        pulse_2_input_measurements |= signals.measure_pulse(input_power_pulse_2, t_eval)
        pulse_2_output_measurements |= signals.measure_pulse(output_power_pulse_2, t_eval)

        pulse_measurements = [
            pulse_1_input_measurements,
            pulse_1_output_measurements,
            pulse_2_input_measurements,
            pulse_2_output_measurements,
        ]

        with open(save_dir / "pulse_1_and_2_info.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pulse_measurements[0].keys())
            writer.writeheader()
            writer.writerows(pulse_measurements)

        # ============================================================
        # Create and save plots
        # ============================================================
        ### Modal gain - losses ###
        plot_modal_gain_minus_alpha(
            n_min=np.min(n),
            n_max=np.max(n),
            wavelength_min=sim_main.spectrum.min,
            wavelength_max=sim_main.spectrum.max,
            modal_gain=soa_material.modal_gain,
            alpha = soa_material.alpha,
            outfile=save_dir / "modal_gain.svg",
        )

        ### Carrier desity ###
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

        ### Forward input and output ###
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

        ### Forward input and output (normalized) ###
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

        ### Forward input and output filtered at pulses wavelengths ###
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



if __name__ == '__main__':
    main()