import numpy as np
from pathlib import Path
from devices import SoaDimensions, SoaConditions, SoaMaterial, SoaSimulationParams
import csv
from collections.abc import Sequence

# Assumes power in W 
# Saves .npz and .csv
def save_pulse_measurements(
    pulse_measurements: Sequence[dict[str, float | str]],
    save_dir: Path | str,
    filename,
    signals_names: Sequence[str] | None = None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    signals_names = complete_signal_names(signals_names, len(pulse_measurements), prefix="pulse")

    pulse_measurements_with_name = [
        {"name": name} | measurement for name, measurement in zip(signals_names, pulse_measurements)
    ]

    fieldnames = pulse_measurements_with_name[0].keys()

    with open(save_dir / f"{filename}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pulse_measurements_with_name)


# Saves .npz and .csv
def save_input_output_power(
    signals_P: Sequence[tuple[np.ndarray, np.ndarray]],
    t: np.ndarray,
    save_dir: Path | str,
    filename,
    signals_names: Sequence[str] | None = None,
    t_label: str = "time_s",
    signals_unit : str = "W",
):
    # signals_P indexing: [n_pulse, input_or_output, t]
    # signals_P shape: (N_pulses, 2, N_t)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    signals_names = complete_signal_names(signals_names, len(signals_P))

    data_dict = {t_label: t}

    for name, signal_P in zip(signals_names, signals_P):
        data_dict[f"{name}_input_{signals_unit}"] = signal_P[0]
        data_dict[f"{name}_output_{signals_unit}"] = signal_P[1]

    data_array = np.column_stack(tuple(data_dict.values()))

    # NPZ
    np.savez(
        save_dir / f"{filename}.npz",
        **data_dict,
    )

    np.savetxt(
        save_dir / f"{filename}.csv",
        data_array,
        delimiter=",",
        comments="",
        header=",".join(data_dict.keys()),
    )

# Saves .npz
# signals_P indexing: [n_pulse, input_or_output, t, wavelength]
# signals_P shape: (N_pulses, 2, N_t, N_spectrum)
def save_input_output_power_spectral(
    signals_P: Sequence[tuple[np.ndarray]],
    t: np.ndarray,
    spectrum: np.ndarray,
    save_dir: Path | str,
    filename,
    signals_names: Sequence[str] | None = None,
    spectrum_label: str = "spectrum_nm",
    t_label: str = "time_s",
    signals_unit : str = "W",
):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    signals_names = complete_signal_names(signals_names, len(signals_P))

    data_dict = {t_label: t, spectrum_label: spectrum}

    for name, signal_P in zip(signals_names, signals_P):
        data_dict[f"{name}_input_{signals_unit}"] = signal_P[0]
        data_dict[f"{name}_output_{signals_unit}"] = signal_P[1]

    # NPZ
    np.savez(
        save_dir / f"{filename}.npz",
        **data_dict,
    )

# Saves .npz
# signals_P indexing: [n_signal, t, z, wavelength]
# signals_P shape: (N_signals, N_t, N_z, N_spectrum)
def save_power_spectral(
    signals_P: Sequence[np.ndarray],
    t: np.ndarray,
    z: np.ndarray,
    spectrum : np.ndarray,
    save_dir: Path | str,
    filename = "P",
    signals_names: Sequence[str] | None = None,
    spectrum_label: str = "spectrum_nm",
    t_label: str = "time_s",
    z_label: str = "z_m",
    signals_unit : str = "W",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    signals_names = complete_signal_names(signals_names, len(signals_P))

    data_dict = {t_label: t, spectrum_label: spectrum, z_label: z}

    for name, signal_P in zip(signals_names, signals_P):
        data_dict[f"{name}_{signals_unit}"] = signal_P

    np.savez(save_dir / f"{filename}.npz", **data_dict)


# Saves .npz
# signals_P indexing: [n_signal, t, z]
# signals_P shape: (N_signals, N_t, N_z)
def save_power(
    signals_P: Sequence[np.ndarray],
    t: np.ndarray,
    z: np.ndarray,
    save_dir: Path | str,
    filename = "P",
    signals_names: Sequence[str] | None = None,
    t_label: str = "time_s",
    z_label: str = "z_m",
    signals_unit : str = "W",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    signals_names = complete_signal_names(signals_names, len(signals_P))

    data_dict = {t_label: t, z_label: z}

    for name, signal_P in zip(signals_names, signals_P):
        data_dict[f"{name}_{signals_unit}"] = signal_P

    np.savez(save_dir / f"{filename}.npz", **data_dict)


def save_n(
    n : np.ndarray, 
    z : np.ndarray, 
    t : np.ndarray,
    save_dir: Path | str,
    filename = "n_sim_result"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        save_dir / f"{filename}.npz",
        t=t,
        z=z,
        n=n,
    )
    np.savetxt(
        save_dir / f"{filename}.csv",
        n,
        delimiter=',',
    )


def save_t_and_z(
    t : np.ndarray,
    z : np.ndarray, 
    save_dir: Path | str,
    filename = "sim_points"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        save_dir / f"{filename}.npz",
        t=t,
        z=z,
    )
    np.savetxt(
        save_dir / f"{filename}_t.csv",
        t,
        delimiter=',',
    )
    np.savetxt(
        save_dir / f"{filename}_z.csv",
        z,
        delimiter=',',
    )

def save_configs(
    soa_conditions: SoaConditions,
    soa_material: SoaMaterial,
    soa_dims : SoaDimensions,
    sim_params : SoaSimulationParams,
    save_dir: Path | str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_outfile = save_dir / "config.toml"
    soa_conditions.to_toml(config_outfile)
    soa_material.to_toml(config_outfile)
    soa_dims.to_toml(config_outfile)
    sim_params.to_toml(config_outfile)


def complete_signal_names(
    names: Sequence[str] | None,
    n_items: int,
    prefix: str = "signal",
) -> tuple[str, ...]:

    names = tuple(names) if names is not None else ()

    if len(names) < n_items:
        print(
            f"Warning: Only {len(names)} names provided for "
            f"{n_items} items. Generating missing names."
        )

    if len(names) > n_items:
        raise ValueError(
            f"More names ({len(names)}) than items ({n_items})."
        )

    return names + tuple(
        f"{prefix}_{i + 1}"
        for i in range(len(names), n_items)
    )