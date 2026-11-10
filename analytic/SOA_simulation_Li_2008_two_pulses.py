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
import devices.SOA_model_Li_2008 as soa
import signals.signals as signals
from spectrum.spectrum import DiscreteSpectrum
from plots import plot_carrier_density, plot_probe_gain_dB, plot_signal_output, plot_modal_gain_minus_alpha

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
class TwoPulsesSoaSimulationParams:
    # Spatial and spectral discretization
    Nz: int = 100
    Nlambda: int = 101

    # Simulation time
    t_start: float = 0.0
    t_end: float = 3e-9

    # Gain spectrum
    spectrum_wavelength_min: float = 1500e-9    #TODO: add better default value
    spectrum_wavelength_max: float = 1600e-9    #TODO: add better default value

    #TODO: implement as list of N pulses
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

    # # Probe
    # probe_wavelength: float = 1580e-9
    # probe_power: float = 40e-9

    # Simulation initial condition
    n_initial : float | np.ndarray = 6.6e24


def create_two_pulses_simulation(
    soa_conditions: soa.SoaConditions,
    soa_material: soa.SoaMaterial,
    soa_dims: soa.SoaDims,
    sim_params: TwoPulsesSoaSimulationParams
):
    # Simulation time span
    t_span = (sim_params.t_start, sim_params.t_end)
    # Spatial and spectral discretization:
    z = np.linspace(0.0, soa_dims.L, sim_params.Nz)
    spectrum = DiscreteSpectrum(
        sim_params.spectrum_wavelength_min,
        sim_params.spectrum_wavelength_max,
        sim_params.Nlambda,
    )

    # Simulation initial condition
    if isinstance(sim_params.n_initial,(float, int)):
        n_initial = np.full_like(z, sim_params.n_initial)
    elif isinstance(sim_params.n_initial,(np.ndarray,)):
        n_initial = np.array(sim_params.n_initial)
    else:
        print("Warning: n_initial for SOA propagation simulation is of unknown type. Using default 6.6e24")
        n_initial = np.full_like(z, 6.6e24)

    soa_sim = soa.SoaSimulation(
        soa_conditions=soa_conditions,
        soa_material=soa_material,
        soa_dims=soa_dims,
        z=z,
        t_span=t_span,
        input_signals_forward_list=[],
        input_signals_forward=lambda _ : np.zeros_like(spectrum.spectrum),
        input_signals_backward_list=[],
        input_signals_backward=lambda _ : np.zeros_like(spectrum.spectrum),
        spectrum=spectrum,
        n_initial=n_initial,
    )

    # Input signals
    ## Pulse 1
    pulse_1_signal = spectrum.monochromatic_signal_to_full_spectrum(
        signal_power=lambda t : signals.gaussian_pulse(
            t,
            power_peak=sim_params.pulse_1_power_peak,
            pulse_width=sim_params.pulse_1_width,
            t_peak=sim_params.pulse_1_t_peak,
        ),
        wavelength=sim_params.pulse_1_wavelength
    )
    ## Pulse 2
    pulse_2_signal = spectrum.monochromatic_signal_to_full_spectrum(
        signal_power=lambda t : signals.gaussian_pulse(
            t,
            power_peak=sim_params.pulse_2_power_peak,
            pulse_width=sim_params.pulse_2_width,
            t_peak=sim_params.pulse_2_t_peak,
        ),
        wavelength=sim_params.pulse_2_wavelength
    )

    soa_sim.add_input_signal(pulse_1_signal, sim_params.pulse_1_forward)
    soa_sim.add_input_signal(pulse_2_signal, sim_params.pulse_2_forward)

    return soa_sim

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_run_name = f"{timestamp}_{run_name}" if run_name else timestamp

    soa_conditions_file = args.soa_conditions_file if args.soa_conditions_file else args.config_file
    soa_material_file = args.soa_material_file if args.soa_material_file else args.config_file
    soa_dims_file = args.soa_dims_file if args.soa_dims_file else args.config_file
    sim_params_file = args.sim_params_file if args.sim_params_file else args.config_file

    # TODO: add cli args for SoaConditions, SoaMaterial and SoaDimensions
    default_soa_conditions=soa.SoaConditions.from_toml(soa_conditions_file,f"{default_case}.soa_conditions")
    
    default_soa_material=soa.SoaMaterial.from_toml(soa_material_file,f"{default_case}.soa_material")
    
    default_soa_dims=soa.SoaDimensions.from_toml(soa_dims_file,f"{default_case}.soa_dimensions")
    
    default_sim_params = TwoPulsesSoaSimulationParams.from_toml(sim_params_file,f"{default_case}.soa_sim_params")
    default_sim_params.update_from_cli_args(args)

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
        sim_params_initial_condition.pulse_1_power_peak = 0.0
        sim_params_initial_condition.pulse_2_power_peak = 0.0

        sim_initial_condition = create_two_pulses_simulation(
            sim_params=sim_params_initial_condition,
            soa_conditions=soa_conditions,
            soa_material=soa_material,
            soa_dims=soa_dims,
        )

        ## Simulate SOA propagation with probe and without inputs to find steady state n(z)
        solution_sim_initial_condition = sim_initial_condition.run()

        #########################################
        # Main simulation                       #
        #########################################

        sim_params_main = copy.copy(sim_params)
        
        # Use steady state n(z) (without input signals) as initial condition for main simulation
        sim_params_main.n_initial = solution_sim_initial_condition.y[:, -1]

        sim_main = create_two_pulses_simulation(
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

        input_signal_backward = sim_main.input_signals_backward
        input_signal_forward = sim_main.input_signals_forward
        z = sim_main.z

        output_power_backward = np.sum(
            [
                soa.propagate_power(
                    input_signal_backward(t), 
                    sim_main.spectrum,
                    n,
                    z,
                    soa_material,
                    soa_dims,
                    forward=False,
                )[0]    # backwards propagated so output is at [0]
                for n, t in zip(n.T, t_eval)
            ],
            axis=1
        )

        if DEBUG:
            stop = time.perf_counter()
            elapsed = stop - start
            print(f"Backward propagation execution time: {elapsed} seconds.")
            start = time.perf_counter()

        output_power_forward = np.sum(
            [
                soa.propagate_power(
                    input_signal_forward(t),
                    sim_main.spectrum,
                    n,
                    z,
                    soa_material,
                    soa_dims,
                    forward=True,
                )[-1]   # forwards propagated so output is at [-1]
                for n, t in zip(n.T, t_eval)
            ],
            axis=1
        )

        if DEBUG:
            stop = time.perf_counter()
            elapsed = stop - start
            print(f"Forward propagation execution time: {elapsed} seconds.")


        ## Probe input power and gain
        input_power_backward = np.array(np.sum([input_signal_backward(t) for t in t_eval], axis=1)) # sum across spectrum to get total power
        # backward_gain_dB = np.array([10*np.log10(p_out/p_in) for p_out, p_in in zip(output_power_backward, input_power_backward)])

        ## Pump input power
        input_power_forward = np.array(np.sum([input_signal_forward(t) for t in t_eval], axis=1)) # sum across spectrum to get total power

        # pump_input_measurements = signals.measure_pulse(input_power_forward, t_eval)
        # pump_output_measurements = signals.measure_pulse(output_power_forward, t_eval)

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
            save_dir / "input_and_output_powers.csv",
                t=t_eval,
                input_power_forward=input_power_forward,
                output_power_forward=output_power_forward,
                input_power_backward=input_power_backward,
                output_power_backward=output_power_backward,
                # probe_gain_dB=backward_gain_dB,
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
                    input_power_forward,
                    output_power_forward,
                    input_power_backward,
                    output_power_backward,
                    # backward_gain_dB,
                )
            ]
        )
        np.savetxt(
            save_dir / "pump_and_probe_powers_and_gain.csv",
            data,
            comments='',
            delimiter=',',
            header="t[s], Input power forward [W], Output power forward [W], Input power backward [W], Output power backward [W], Probe gain [dB]"
        )


        # TODO: measure pulses characteristics. Maybe filtering by wavelength?
        # pump_input_measurements["name"] = "input"
        # pump_output_measurements["name"] = "output"

        # pump_measurements = [
        #     pump_input_measurements,
        #     pump_output_measurements,
        # ]

        # with open(save_dir / "pump_pulse_info.csv", "w", newline="") as f:
        #     writer = csv.DictWriter(f, fieldnames=pump_measurements[0].keys())
        #     writer.writeheader()
        #     writer.writerows(pump_measurements)

        # ============================================================
        # Create and save plots
        # ============================================================
        ### Modal gain - losses ###
        plot_modal_gain_minus_alpha(
            n_min=np.min(n),
            n_max=np.max(n),
            wavelength_min=sim_main.spectrum.min,
            wavelength_max=sim_main.spectrum.max,
            modal_gain=lambda n, wavelength : soa_material.modal_gain(n, wavelength),
            alpha = soa_material.alpha,
            outfile=save_dir / "modal_gain.svg",
        )

        ### Carrier desity ###
        plot_carrier_density(
            t_eval,
            z,
            n,
            outfile=save_dir / "carrier_density.svg",
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
            outfile=save_dir / "forward_input_output.svg",
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
            outfile=save_dir / "forward_input_output_normalized.svg",
            show=False,
        )


if __name__ == '__main__':
    main()