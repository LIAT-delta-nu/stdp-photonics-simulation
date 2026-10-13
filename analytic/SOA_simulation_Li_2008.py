import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass, astuple

import devices.SOA_model_Li_2008 as soa
import signals.signals as signals

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
    t_eval: tuple[float]


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
            rtol=1e-5,
            atol=1e18,
        )

        fig, ax = plt.subplots()
        ax.plot(steady.t, steady.y[0,:], label="z=0")
        ax.plot(steady.t, steady.y[-1,:], label="z=L")
        ax.legend()
        ax.grid()


        # Use steady state as initial condition for main simulation
        n_steady_state = steady.y[:, -1]

        # Main simulation
        solution = solve_ivp(
            fun=soa.dn_dt,
            t_span=(self.t_eval[0], self.t_eval[-1]),
            y0=n_steady_state,    # start from steady state
            t_eval=self.t_eval,
            args=(
                self.conditions,
                self.material,
                self.dims,
                self.input_signals_forward,
                self.input_signals_backward,
                z
            ),
            method="BDF",
            rtol=1e-5,
            atol=1e18,
        )
        return solution


def main():
    # ============================================================
    # Calculate observable gain versus time
    # ============================================================

    # Input signals
    ## Pump
    lambda_pump = 1551e-9
    pump_power_peak = 5e-3
    pump_pulse_width = 65e-12   # FWHM
    pump_t_peak = 200e-12

    ## Probe
    lambda_probe = 1550e-9
    probe_power = 40e-6     # W

    # Simulation params
    Nz = 100

    Nt = 1000
    t_start = 0.0
    t_end = 2.0e-9
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
        power = lambda t: probe_power
    )


    # Simulation object
    sim = Simulation(
        conditions=soa.SoaConditions.from_toml("./sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_conditions"),
        material=soa.SoaMaterial.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_material"),
        dims=soa.SoaDimensions.from_toml("sim_configs/SOA_simulation_Li_2008.toml","case_a.soa_dimensions"),
        input_signals_backward=(probe_signal,),
        input_signals_backward_steady_state=(probe_signal,),
        input_signals_forward=(pump_signal,),
        input_signals_forward_steady_state=(),
        n_initial_guess=3e24,   # TODO: change n initial guess in a case by case basis
        Nz=Nz,
        t_eval=t_eval
    )

    for case in CASES:
        sim.conditions.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_conditions")
        sim.material.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_material")
        sim.dims.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_dimensions")
        # sim.n_initial_guess = ... # TODO: change n initial guess in a case by case basis

        solution = sim.run()

        probe_output = np.empty_like(t_eval)
        # pump_output = np.empty_like(t_eval)
        # pump_gain = np.empty_like(t_eval)


        for j, t in enumerate(t_eval):

            n = solution.y[:, j]

            # pump_propagation = soa.propagate_power(
            #     pump_signal.power(t),
            #     n,
            #     pump_signal.lamda,
            #     sim.z(),
            #     sim.material,
            #     sim.dims,
            #     forward=True,
            # )

            probe_propagation = soa.propagate_power(
                probe_signal.power(t),
                n,
                probe_signal.lamda,
                sim.z(),
                sim.material,
                sim.dims,
                forward=False,
            )

            # pump_output[j] = pump_propagation[-1]
            probe_output[j] = probe_propagation[0]

            # pump_gain[j] = ( pump_output[j]) / ( pump_signal.power(t) + 1e-50 )


        probe_gain = probe_output/probe_power
        # pump_gain_dB = 10.0 * np.log10(pump_gain)
        probe_gain_dB = 10.0 * np.log10(probe_gain)


        # ============================================================
        # Plot carrier density
        # ============================================================

        plt.figure()

        plt.imshow(
            solution.y,
            aspect="auto",
            origin="lower",
            extent=[
                t_eval[0] * 1e12,
                t_eval[-1] * 1e12,
                0,
                sim.dims.L * 1e6,
            ],
        )

        plt.xlabel("Time (ps)")
        plt.ylabel("Position z (um)")
        plt.colorbar(label="Carrier density (m$^{-3}$)")

        plt.title("SOA carrier-density dynamics")


        # # ============================================================
        # # Plot pump 
        # # ============================================================

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

        plt.figure()

        plt.plot(
            t_eval * 1e12,
            probe_gain_dB,
        )

        plt.xlabel("Time (ps)")
        plt.ylabel("Probe gain (dB)")

        plt.title(f"SOA gain depletion and recovery (probe at {lambda_probe*1e9:5.1f} nm)")

        plt.grid()


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

        plt.show()

if __name__ == '__main__':
    main()