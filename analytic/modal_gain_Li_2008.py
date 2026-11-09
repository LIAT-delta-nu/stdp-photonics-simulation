import numpy as np
import matplotlib.pyplot as plt
from devices.SOA_model_Li_2008 import SoaMaterial
from plots import plot_modal_gain_minus_alpha_mesh, plot_modal_gain_minus_alpha_sign_mesh


DEFAULT_CASE = "case_a"
CASES = (
    "case_a", 
    # "case_b", 
    # "case_c", 
    # "case_d",
)

def modal_gain_mesh(
        n_center: float,
        lambda_center: float,
        lambda_sweep_width: float = 50e-9,
        n_sweep_width_decades: int = 4,
        lambda_sweep_n : int = 100,
        n_sweep_n : int = 100,
):
    log10_n = np.log10(n_center)
    ns = np.logspace(log10_n - n_sweep_width_decades / 2, log10_n + n_sweep_width_decades / 2, lambda_sweep_n)
    lambdas = np.linspace(lambda_center - lambda_sweep_width / 2, lambda_center + lambda_sweep_width / 2, n_sweep_n)

    N, Lambdas = np.meshgrid(ns, lambdas)

    return N, Lambdas



def main():

    material = SoaMaterial.from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{DEFAULT_CASE}.soa_material")

    for case in CASES:
        print(case)

        material.update_from_toml("sim_configs/SOA_simulation_Li_2008.toml",f"{case}.soa_material")

        N, Lambdas = modal_gain_mesh(
            n_center=5.1e+24,
            n_sweep_width_decades=1,
            lambda_center=1.55e-6,
            lambda_sweep_width=0.1e-6
        )

        M = material.modal_gain(N, Lambdas) - material.alpha

        plot_modal_gain_minus_alpha_mesh(N, Lambdas, M)


    plt.show()


if __name__ == '__main__':
    main()