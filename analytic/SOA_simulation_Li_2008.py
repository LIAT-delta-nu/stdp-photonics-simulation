import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass, astuple
from pathlib import Path

import devices.SOA_model_Li_2008 as soa
import signals.signals as signals
from plots import plot_carrier_density, plot_probe_gain_dB

CASES = (
    "case_a", 
    # "case_aa", 
    # "case_ab", 
    "case_b", 
    "case_c", 
    "case_d",
)

@dataclass
class Simulation():
    conditions: soa.SoaConditions
    material: soa.SoaMaterial
    dims: soa.SoaDimensions
    input_signals_forward: tuple[soa.SoaInputSignal]
    input_signals_forward_steady_state: tuple[soa.SoaInputSignal]
    input_signals_backward: tuple[soa.SoaInputSignal]
    input_signals_backward_steady_state: tuple[soa.SoaInputSignal]
    n_initial_guess: float
    Nz: int
    t_span: tuple[float]


    def z(self):
        return np.linspace(0.0, self.dims.L, self.Nz)

    def run(self):
        z = self.z()

        # Initial guess to calculate n(z) in steady state
        n_initial_guess = np.full(z.shape, self.n_initial_guess)

        # Find approximately steady state
        steady = solve_ivp(
            fun=soa.dn_dt,
            t_span=(0, 1e-3),
            y0=n_initial_guess,
            args=(
                self.conditions,
                self.material,
                self.dims,
                self.input_signals_forward_steady_state,
                self.input_signals_backward_steady_state,
                z
            ),
            method="BDF",
            rtol=1e-10,
            atol=1e0,
        )

        # Use steady state as initial condition for main simulation
        n_steady_state = steady.y[:, -1]

        # Main simulation
        solution = solve_ivp(
            fun=soa.dn_dt,
            t_span=self.t_span,
            y0=n_steady_state,    # start from steady state
            # t_eval=self.t_eval,
            args=(
                self.conditions,
                self.material,
                self.dims,
                self.input_signals_forward,
                self.input_signals_backward,
                z
            ),
            method="BDF",
            rtol=1e-10,
            atol=1e0,
        )
        return solution


def main():
    # ============================================================
    # Calculate observable gain versus time
    # ============================================================

    # Input signals
    ## Pump
    lambda_pump = 1551e-9
    pump_power_peak = 0
    # pump_power_peak = 5e-3
    pump_pulse_width = 65e-12   # FWHM
    pump_t_peak = 200e-12

    ## Probe
    lambda_probe = 1580e-9
    probe_power_input = 40e-6     # W

    # Simulation params
    Nz = 100

    Nt = 1000
    t_start = 0.0
    t_end = 3e-9
    t_eval = np.linspace(
        t_start,
        t_end,
        Nt
    )

    # Input signals 
    ## Pump (forward)
    pump_signal : soa.SoaInputSignal = soa.SoaInputSignal(
        lamda = lambda_pump,
        power = lambda t: signals.gaussian_pulse(t, pump_power_peak, pump_pulse_width, pump_t_peak)
    )

    ## Probe (backwards)
    probe_signal : soa.SoaInputSignal = soa.SoaInputSignal(
        lamda = lambda_probe,
        power = lambda t: probe_power_input
    )


    # Simulation object
    sim = Simulation(
        conditions=soa.SoaConditions.from_toml("./sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_conditions"),
        material=soa.SoaMaterial.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_material"),
        dims=soa.SoaDimensions.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_dimensions"),
        input_signals_backward=(probe_signal,),
        input_signals_backward_steady_state=(probe_signal,),
        input_signals_forward=(pump_signal,),
        # input_signals_forward=(),
        input_signals_forward_steady_state=(),
        n_initial_guess=3e24,   # TODO: change n initial guess in a case by case basis
        Nz=Nz,
        t_span=(t_start, t_end),
    )



    for case in CASES:
        print(case)
        sim.conditions.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_conditions")
        sim.material.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_material")
        sim.dims.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_dimensions")
        # sim.n_initial_guess = ... # TODO: change n initial guess in a case by case basis
        z = sim.z()

        # sim.dims.d = d

        solution = sim.run()

        # Calc. probe and pump output

        t_eval = solution.t

        probe_output = [
                    soa.propagate_power(
                    probe_signal.power(t),
                    n,
                    probe_signal.lamda,
                    z,
                    sim.material,
                    sim.dims,
                    forward=False,
                )[0]
            for n, t in zip(solution.y.T, t_eval)
        ]

        pump_output = [
                    soa.propagate_power(
                    pump_signal.power(t),
                    n,
                    pump_signal.lamda,
                    z,
                    sim.material,
                    sim.dims,
                    forward=True,
                )[0]
            for n, t in zip(solution.y.T, t_eval)
        ]


        # Probe gain
        probe_gain_dB = [10*np.log10(p/probe_signal.power(t)) for p, t in zip(probe_output, t_eval)]
        # pump_gain_dB = [10*np.log10(p/pump_signal.power(t)) for p, t in zip(pump_output, t_eval)]

        # ============================================================
        # Plots
        # ============================================================

        plot_carrier_density(
            t_eval,
            z,
            solution.y
        )
        plot_probe_gain_dB(
            t_eval, 
            probe_gain_dB
        )

        # plt.plot(
        #     lambdas*1e9,
        #     probe_gain_db,
        #     label=f"{case}"
        # )

        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Probe gain (dB)")

        # # plt.title(f"SOA gain depletion and recovery (probe at {lambda_probe*1e9:5.1f} nm)")

        # plt.grid()




        # # ============================================================
        # # Plot pump 
        # # ============================================================

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

            # pump_gain[j] = ( pump_output[j]) / ( pump_signal.power(t) + 1e-50 )



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