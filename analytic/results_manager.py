import numpy as np
from pathlib import Path
from devices import SoaDimensions, SoaConditions, SoaMaterial, SoaSimulationParams, SoaSimulationResult
import csv
from collections.abc import Sequence

###
### Pulse measurements
###

def load_pulse_measurements(
    save_dir: Path | str,
    filename: str,
) -> list[dict[str, float | str]]:

    save_dir = Path(save_dir)
    filename = Path(filename).with_suffix(".csv")
    
    with open(save_dir / f"{filename}", "r", newline="") as f:
        reader = csv.DictReader(f)

        pulse_measurements = [
            {
                key: (value if key == "name" else float(value))
                for key, value in row.items()
            }
            for row in reader
        ]

    return pulse_measurements

def save_pulse_measurements(
    pulse_measurements: dict[str, dict[str, float | str]],
    save_dir: Path | str,
    filename,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pulse_measurements_with_name = [
        {"name": single_pulse_name} | single_pulse_measurements
        for single_pulse_name, single_pulse_measurements in pulse_measurements.items()
    ]

    fieldnames = pulse_measurements_with_name[0].keys()

    with open(save_dir / f"{filename}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pulse_measurements_with_name)

###
### Input and output powers (no wavelength, no z inside SOA)
###
def load_input_output_power(
    save_dir: Path | str,
    filename: str,
    t_label: str = "time_s",
    signals_unit: str = "W",
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:

    save_dir = Path(save_dir)
    filename = Path(filename).stem

    npz_file = save_dir / f"{filename}.npz"
    csv_file = save_dir / f"{filename}.csv"

    # Load data
    if npz_file.exists():
        with np.load(npz_file) as data:
            data_dict = {key: data[key] for key in data.files}

    elif csv_file.exists():
        data_array = np.genfromtxt(
            csv_file,
            delimiter=",",
            names=True,
            encoding="utf-8",
        )
        data_dict = {
            name: data_array[name]
            for name in data_array.dtype.names
        }

    else:
        raise FileNotFoundError(
            f"Neither {npz_file} nor {csv_file} exists."
        )

    # Extract time
    t = data_dict.pop(t_label)

    # Extract input/output signals
    suffix_input = f"_input_{signals_unit}"
    suffix_output = f"_output_{signals_unit}"

    signal_names = [
        key.removesuffix(suffix_input)
        for key in data_dict
        if key.endswith(suffix_input)
    ]

    signals_P = [
        (
            data_dict[f"{name}{suffix_input}"],
            data_dict[f"{name}{suffix_output}"],
        )
        for name in signal_names
    ]

    return t, signals_P

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


###
### Input and output powers spectral (no z inside SOA)
###

def load_input_output_power_spectral(
    save_dir: Path | str,
    filename: str,
    spectrum_label: str = "spectrum_nm",
    t_label: str = "time_s",
    signals_unit: str = "W",
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:

    save_dir = Path(save_dir)
    filename = Path(filename).stem
    npz_file = save_dir / f"{filename}.npz"

    if not npz_file.exists():
        raise FileNotFoundError(npz_file)

    with np.load(npz_file) as data:
        t = data[t_label]
        spectrum = data[spectrum_label]

        suffix_input = f"_input_{signals_unit}"
        suffix_output = f"_output_{signals_unit}"

        signal_names = [
            key.removesuffix(suffix_input)
            for key in data.files
            if key.endswith(suffix_input)
        ]

        signals_P = [
            (
                data[f"{name}{suffix_input}"],
                data[f"{name}{suffix_output}"],
            )
            for name in signal_names
        ]

    return t, spectrum, signals_P

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

###
### Powers spectral
###

def load_power_spectral(
    save_dir: Path | str,
    filename: str = "P",
    spectrum_label: str = "spectrum_nm",
    t_label: str = "time_s",
    z_label: str = "z_m",
    signals_unit: str = "W",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:

    save_dir = Path(save_dir)
    filename = Path(filename).stem
    npz_file = save_dir / f"{filename}.npz"

    if not npz_file.exists():
        raise FileNotFoundError(npz_file)

    with np.load(npz_file) as data:
        t = data[t_label]
        z = data[z_label]
        spectrum = data[spectrum_label]

        suffix = f"_{signals_unit}"

        signals_P = [
            data[key].copy()
            for key in data.files
            if key.endswith(suffix)
        ]

    return t, z, spectrum, signals_P

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

###
### Powers
###

def load_power(
    save_dir: Path | str,
    filename: str = "P",
    t_label: str = "time_s",
    z_label: str = "z_m",
    signals_unit: str = "W",
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:

    save_dir = Path(save_dir)
    filename = Path(filename).stem
    npz_file = save_dir / f"{filename}.npz"

    if not npz_file.exists():
        raise FileNotFoundError(npz_file)

    with np.load(npz_file) as data:
        t = data[t_label]
        z = data[z_label]

        suffix = f"_{signals_unit}"

        signals_P = [
            data[key].copy()
            for key in data.files
            if key.endswith(suffix)
        ]

    return t, z, signals_P

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

###
### Carrier concentration
###

def load_n(
    save_dir: Path | str,
    filename: str = "n_sim_result",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:

    save_dir = Path(save_dir)
    filename = Path(filename).stem

    npz_file = save_dir / f"{filename}.npz"
    csv_file = save_dir / f"{filename}.csv"

    if npz_file.exists():
        with np.load(npz_file) as data:
            n = data["n"].copy()
            z = data["z"].copy()
            t = data["t"].copy()

    elif csv_file.exists():
        n = np.loadtxt(csv_file, delimiter=",")
        z = None
        t = None

    else:
        raise FileNotFoundError(
            f"Neither {npz_file} nor {csv_file} exists."
        )

    return n, z, t

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

###
### t and z
###

def load_t_and_z(
    save_dir: Path | str,
    filename: str = "sim_points",
) -> tuple[np.ndarray, np.ndarray]:

    save_dir = Path(save_dir)
    filename = Path(filename).stem

    npz_file = save_dir / f"{filename}.npz"
    csv_t_file = save_dir / f"{filename}_t.csv"
    csv_z_file = save_dir / f"{filename}_z.csv"

    if npz_file.exists():
        with np.load(npz_file) as data:
            t = data["t"].copy()
            z = data["z"].copy()

    elif csv_t_file.exists() and csv_z_file.exists():
        t = np.loadtxt(csv_t_file, delimiter=",")
        z = np.loadtxt(csv_z_file, delimiter=",")

    else:
        raise FileNotFoundError(
            f"Could not find either {npz_file} or both CSV files "
            f"({csv_t_file}, {csv_z_file})."
        )

    return t, z

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


###
### SoaSimulationResult
###

# Saves .npz
def save_soa_sim_result(
    result : SoaSimulationResult,
    save_dir: Path | str,
    t_label: str = "t_s",
    z_label: str = "z_m",
    n_label: str = "n_m-3",
    spectrum_label: str = "spectrum_nm",
    P_forward_label: str = "P_forward_W",
    P_backward_label: str = "P_backward_W",
    filename = "soa_sim_results.npz",
):
    data_dict = {
        t_label: result.t, 
        z_label: result.z, 
        n_label: result.n, 
        spectrum_label: result.spectrum, 
        P_forward_label: result.P_forward,
        P_backward_label: result.P_backward,
    }
    np.savez(
        save_dir / f"{filename}",
        **data_dict,
    )


def load_soa_sim_result(
    save_dir: Path | str,
    t_label: str = "t_s",
    z_label: str = "z_m",
    n_label: str = "n_m-3",
    spectrum_label: str = "spectrum_nm",
    P_forward_label: str = "P_forward_W",
    P_backward_label: str = "P_backward_W",
    filename: str = "soa_sim_results.npz",
) -> SoaSimulationResult:

    with np.load(Path(save_dir) / filename) as data:

        return SoaSimulationResult(
            t=data[t_label],
            z=data[z_label],
            n=data[n_label],
            spectrum=data[spectrum_label],
            P_forward=data[P_forward_label],
            P_backward=data[P_backward_label],
        )

###
### configs
###

# TODO: load_configs.

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


def create_processed_dir(parent_dir: Path | str) -> Path:

    parent_dir = Path(parent_dir)

    processed_dir = parent_dir / "processed"

    i = 1
    while processed_dir.exists():
        processed_dir = parent_dir / f"processed_{i:02d}"
        i += 1

    processed_dir.mkdir()

    return processed_dir