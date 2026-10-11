import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import SOA.SOA_model_Li_2008 as soa
import signals.signals as signals


def main():

    # SOA dims
    L = 800e-6                # 800 um
    W = 2.0e-6                # active-region width [m]
    d = 0.2e-6                # active-region thickness [m]

    # SOA injection current
    I = 72e-3                 # 72 mA

    # Input signals
    lambda_pump = 1551e-9
    lambda_probe = 1550e-9

    pump_power_peak = 5e-3
    pump_pulse_width = 65e-12   # FWHM
    pump_t_peak = 200e-12

    probe_input_power = 40e-6     # W

    # Simulation params
    Nz = 100
    z = np.linspace(0.0, L, Nz)

    t_start = 0.0
    t_end = 2.0e-9
    t_eval = np.linspace(
        t_start,
        t_end,
        1000
    )

    condition = soa.SoaCondition(I=87e-3)
    material = soa.SoaMaterial()
    dims = soa.SoaDimensions(W, d, L)

    pump_signal : soa.SoaInputSignal = soa.SoaInputSignal(
        lamda = lambda_pump,
        power = lambda t: signals.gaussian_pulse(t, pump_power_peak, pump_pulse_width, pump_t_peak)
    )

    probe_signal : soa.SoaInputSignal = soa.SoaInputSignal(
        lamda = lambda_probe,
        power = lambda t: probe_input_power
    )

    input_signals = (pump_signal, probe_signal,)


    # ============================================================
    # Dynamic simulation
    # ============================================================

    # Initial guess to calculate steady state
    n_initial = np.full(Nz, 2.7e26)

    # Find approximately steady state
    steady = solve_ivp(
        fun=soa.dn_dt,
        t_span=(0.0, 100e-9),
        y0=n_initial,
        args=(
            condition,
            material,
            dims,
            (probe_signal,),
            z
        ),
        method="BDF",
        rtol=1e-5,
        atol=1e18,
    )

    print(steady.message)
    print(steady.y[-1,:])

    # Use steady state as initial condition for main simulation
    n_ss = steady.y[:, -1]
    # print(n_ss)
    solution = solve_ivp(
        fun=soa.dn_dt,
        t_span=(t_start, t_end),
        y0=n_ss,    # start from steady state
        t_eval=t_eval,
        args=(
            condition,
            material,
            dims,
            input_signals,
            z
        ),
        method="BDF",
        rtol=1e-5,
        atol=1e18,
    )


    # ============================================================
    # Calculate observable gain versus time
    # ============================================================

    probe_output = np.empty(len(t_eval))
    pump_output = np.empty(len(t_eval))

    pump_gain = np.zeros_like(pump_output)

    for j, t in enumerate(t_eval):

        n = solution.y[:, j]

        Pp = soa.propagate_power(
            pump_signal.power(t),
            n,
            pump_signal.lamda,
            z,
            material,
            dims
        )

        Ps = soa.propagate_power(
            probe_signal.power(t),
            n,
            probe_signal.lamda,
            z,
            material,
            dims
        )

        pump_output[j] = Pp[-1]
        probe_output[j] = Ps[-1]

        pump_gain[j] = ( pump_output[j]) / ( pump_signal.power(t) + 1e-50 )


    probe_gain = probe_output/probe_input_power
    pump_gain_dB = 10.0 * np.log10(pump_gain)
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
            L * 1e6,
        ],
    )

    plt.xlabel("Time (ps)")
    plt.ylabel("Position z (um)")
    plt.colorbar(label="Carrier density (m$^{-3}$)")

    plt.title("SOA carrier-density dynamics")


    # ============================================================
    # Plot pump 
    # ============================================================

    # Pump output
    plt.figure()

    plt.plot(
        t_eval * 1e12,
        pump_output * 1e3,
    )

    plt.xlabel("Time (ps)")
    plt.ylabel("Pump output power (mW)")

    plt.title("Pump output")

    # Pump input and output power (normalized)
    
    pump_input = np.array([pump_signal.power(t) for t in t_eval])
    pump_input_max = pump_power_peak
    pump_output_max = np.max(pump_output)

    plt.figure()
    plt.plot(
        t_eval * 1e12,
        pump_input/pump_input_max,
        label="Input"
    )
    plt.plot(
        t_eval * 1e12,
        pump_output/pump_output_max,
        label="Output"
    )

    plt.legend()

    plt.xlabel("Time (ps)")
    plt.ylabel("Pump power (a.u.)")

    plt.title("Pump power")


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


    # ============================================================
    # Plot pump gain
    # ============================================================

    plt.figure()

    plt.plot(
        t_eval * 1e12,
        pump_gain_dB,
    )

    plt.xlabel("Time (ps)")
    plt.ylabel("Pump gain (dB)")

    plt.title(f"SOA gain depletion and recovery (pump at {lambda_pump*1e9:5.1f} nm)")

    plt.grid()
    # ============================================================
    # Plot probe gain
    # ============================================================

    plt.figure()

    plt.plot(
        t_eval * 1e12,
        probe_gain_dB,
        label="probe",
    )

    plt.plot(
        t_eval * 1e12,
        pump_gain_dB,
        label="pump",
    )

    plt.legend()
    plt.xlabel("Time (ps)")
    plt.ylabel("Gain (dB)")

    plt.title(f"SOA gain depletion and recovery (probe at {lambda_probe*1e9:5.1f} nm and pump at {lambda_pump*1e9:5.1f} nm)")

    plt.grid()

    plt.show()

if __name__ == '__main__':
    main()