VERBOSE = True
OUTDIR = "./out"


import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
if VERBOSE:
    import time

from stdp_photonics.config import toml_config, cli_config
import copy
from stdp_photonics.devices import (
    SoaSimulationParams, 
    SoaMaterial, 
    SoaConditions, 
    SoaSimulation, 
    SoaDimensions, 
    soa_propagate_power
)
# from stdp_photonics.devices.SOA_model_Li_2008 import soa_propagate_power
import stdp_photonics.signals.signals as signals
from stdp_photonics.spectrum.spectrum import DiscreteSpectrum
from stdp_photonics.plots import plot_carrier_density, plot_probe_gain_dB, plot_signal_output, plot_modal_gain_minus_alpha

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
class PumpProbeSoaSimulationParams(SoaSimulationParams):
    sim_type_name : str = "pump_probe"

    # Pump
    pump_wavelength: float = 1551e-9
    pump_power_peak: float = 5e-3
    pump_pulse_width: float = 65e-12
    pump_t_peak: float = 200e-12

    # Probe
    probe_wavelength: float = 1580e-9
    probe_power: float = 40e-9

    @property
    def input_signals_forward_list(self):
        # Pump (forward)
        ## Note: local vars are used to force lambda to bind to the local vars by value and not to self by reference
        ## Therefore, if self.power_peak, self.pulse_width, etc. changes, the lambda stays the same
        pump_power_peak=self.pump_power_peak
        pump_pulse_width=self.pump_pulse_width
        pump_t_peak=self.pump_t_peak
        pump_wavelength=self.pump_wavelength
        pump_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
            signal_power=lambda t : signals.gaussian_pulse(
                t,
                power_peak=pump_power_peak,
                pulse_width=pump_pulse_width,
                t_peak=pump_t_peak,
            ),
            spectral_position=pump_wavelength
        )
        return [pump_signal]

    @property
    def input_signals_backward_list(self):
        # Probe (backwards)
        ## Note: local vars are used to force lambda to bind to the local vars by value and not to self by reference
        ## Therefore, if self.probe_power, etc. changes, the lambda stays the same
        probe_power=self.probe_power
        pump_wavelength=self.pump_wavelength
        probe_signal = self.spectrum.monochromatic_signal_to_full_spectrum(
            signal_power=lambda _ : probe_power,
            spectral_position=pump_wavelength
        )
        return [probe_signal]

def build_parser():
    parser = ArgumentParser()
    PumpProbeSoaSimulationParams.add_cli_args_to_parser(parser)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    soa_conditions_file = args.soa_conditions_file if args.soa_conditions_file else args.config_file
    soa_material_file = args.soa_material_file if args.soa_material_file else args.config_file
    soa_dims_file = args.soa_dims_file if args.soa_dims_file else args.config_file
    sim_params_file = args.sim_params_file if args.sim_params_file else args.config_file

    # TODO: add cli args for SoaConditions, SoaMaterial and SoaDimensions
    default_soa_conditions=SoaConditions.from_toml(soa_conditions_file,f"{default_case}.soa_conditions")
    
    default_soa_material=SoaMaterial.from_toml(soa_material_file,f"{default_case}.soa_material")
    
    default_soa_dims=SoaDimensions.from_toml(soa_dims_file,f"{default_case}.soa_dimensions")
    
    default_sim_params = PumpProbeSoaSimulationParams.from_toml(sim_params_file,f"{default_case}.soa_sim_params")
    default_sim_params.update_from_cli_args(args)

    sim_type_name = default_sim_params.sim_type_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_run_name = f"{timestamp}__{sim_type_name}__{run_name}" if run_name else f"{timestamp}__{sim_type_name}"

    cases = list(dict.fromkeys(cases))  # Remove duplicates, preserving order

    for case in cases:
        if VERBOSE:
            print(case)

        # First, simulate SOA propagation of only the probe to find the initial condition of the main simulation
        # Second, do the main simulation (pump and probe) using use the steady state of the probe-only simulation 
        # as initial n(z) 

        soa_conditions = copy.copy(default_soa_conditions)
        soa_material = copy.copy(default_soa_material)
        soa_dims = copy.copy(default_soa_dims)
        sim_params =  copy.copy(default_sim_params)

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
        sim_params_initial_condition.pump_power_peak = 0.0

        sim_initial_condition = SoaSimulation(
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

        # Calc. probe and pump output
        n = solution_sim_main.y
        t_eval = solution_sim_main.t

        if VERBOSE:
            start_probe = time.perf_counter()

        probe_signal = sim_main.input_signal_backward # probe is propagated backwards
        pump_signal = sim_main.input_signal_forward # pump is propagated forward
        z = sim_main.z

        probe_output_power = np.sum(
            [
                soa_propagate_power(
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

        if VERBOSE:
            stop_probe = time.perf_counter()
            elapsed_probe = stop_probe - start_probe
            print(f"Probe power propagation execution time: {elapsed_probe} seconds.")
            start_pump = time.perf_counter()

        pump_output_power = np.sum(
            [
                soa_propagate_power(
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

        if VERBOSE:
            stop_pump = time.perf_counter()
            elapsed_pump = stop_pump - start_pump
            print(f"Pump power propagation execution time: {elapsed_pump} seconds.")


        ## Probe input power and gain
        probe_input_power = np.array(np.sum([probe_signal(t) for t in t_eval], axis=1)) # sum across spectrum to get total power
        probe_gain_dB = np.array([10*np.log10(p_out/p_in) for p_out, p_in in zip(probe_output_power, probe_input_power)])

        ## Pump input power
        pump_input_power = np.array(np.sum([pump_signal(t) for t in t_eval], axis=1)) # sum across spectrum to get total power

        pump_input_measurements = signals.measure_pulse(pump_input_power, t_eval)
        pump_output_measurements = signals.measure_pulse(pump_output_power, t_eval)

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

        pump_input_measurements["name"] = "input"
        pump_output_measurements["name"] = "output"

        pump_measurements = [
            pump_input_measurements,
            pump_output_measurements,
        ]

        with open(save_dir / "pump_pulse_info.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pump_measurements[0].keys())
            writer.writeheader()
            writer.writerows(pump_measurements)

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

        ### Probe gain dB ###
        plot_probe_gain_dB(
            t_eval, 
            probe_gain_dB=probe_gain_dB,
            signal_power_input=pump_input_power,
            signal_power_output=pump_output_power,
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


if __name__ == '__main__':
    main()