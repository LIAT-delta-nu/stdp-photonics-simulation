
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import copy
import numpy as np

from devices import (
    SoaConditions,
    SoaMaterial,
    SoaDimensions,
    SoaSimulationParams,
    SoaSimulationResult,
)

from results_manager import load_soa_sim_result

@dataclass
class ExperimentSetup:

    full_run_name: str
    timestamp: str
    sim_type_name: str

    cases: list[str]
    default_case: str

    soa_conditions: dict[str, SoaConditions]
    soa_material: dict[str, SoaMaterial]
    soa_dims: dict[str, SoaDimensions]
    sim_params: dict[str, SoaSimulationParams]

    sim_data_dir : Path | str | None
    process : bool

    @classmethod
    def from_args(
        cls,
        args,
        sim_params_class: type[SoaSimulationParams],
        cases_if_missing: list[str],
        timestamp=None,
    ) -> ExperimentSetup:

        sim_data_dir = args.sim_data_dir
        process = not args.no_processing

        # Cases
        cases = list(dict.fromkeys(args.cases or cases_if_missing))

        if not cases:
            raise ValueError("No simulation cases specified.")

        # Configuration files
        config_file = args.config_file

        soa_conditions_file = args.soa_conditions_file or config_file
        soa_material_file = args.soa_material_file or config_file
        soa_dims_file = args.soa_dims_file or config_file
        sim_params_file = args.sim_params_file or config_file

        soa_conditions = {}
        soa_material = {}
        soa_dims = {}
        sim_params = {}

        default_case = args.default_case or cases[0]

        default_soa_conditions = SoaConditions.from_toml(
            soa_conditions_file, f"{default_case}.soa_conditions"
        )

        default_soa_material = SoaMaterial.from_toml(
            soa_material_file, f"{default_case}.soa_material"
        )

        default_soa_dims = SoaDimensions.from_toml(
            soa_dims_file, f"{default_case}.soa_dimensions"
        )

        default_sim_params = sim_params_class.from_toml(
            sim_params_file, f"{default_case}.soa_sim_params"
        )

        # Load all cases:
        # 1. Load default values
        # 2. Override with any case-specific modification
        # 3. Override with any cli parameter
        for case in cases:
            # Apply case-specific TOML overrides
            soa_conditions[case] = copy.copy(
                default_soa_conditions
            )
            soa_conditions[case].update_from_toml(
                soa_conditions_file, f"{case}.soa_conditions"
            ).update_from_cli_args(args)

            soa_material[case] = copy.copy(
                default_soa_material
            )
            soa_material[case].update_from_toml(
                soa_material_file, f"{case}.soa_material"
            ).update_from_cli_args(args)

            soa_dims[case] = copy.copy(
                default_soa_dims
            )
            soa_dims[case].update_from_toml(
                soa_dims_file, f"{case}.soa_dimensions"
            ).update_from_cli_args(args)

            sim_params[case] = copy.copy(
                default_sim_params
            )
            sim_params[case].update_from_toml(
                sim_params_file, f"{case}.soa_sim_params"
            ).update_from_cli_args(args)


        soa_conditions["default"] = soa_conditions[default_case]
        soa_material["default"] = soa_material[default_case]
        soa_dims["default"] = soa_dims[default_case]
        sim_params["default"] = sim_params[default_case]

        # Timestamp
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        sim_type_name=sim_params["default"].sim_type_name

        # Full run name
        if sim_data_dir:
            full_run_name = Path(sim_data_dir).name
        elif args.run_name:
            full_run_name = f"{timestamp}__{sim_type_name}__{args.run_name}"
        else:
            full_run_name = f"{timestamp}__{sim_type_name}"

        return cls(
            full_run_name=full_run_name,
            timestamp=timestamp,
            sim_type_name=sim_type_name,
            sim_data_dir=sim_data_dir,
            process=process,
            cases=cases,
            default_case=default_case,
            soa_conditions=soa_conditions,
            soa_material=soa_material,
            soa_dims=soa_dims,
            sim_params=sim_params,
        )

    def update_from_dir(
        self, 
        sim_params_class: type[SoaSimulationParams],
        filename : str = 'config.toml',
    ):
        load_sim_and_soa_configs_from_dir(
            self.sim_data_dir, 
            sim_params_class,
            self.sim_params,
            self.soa_conditions,
            self.soa_material,
            self.soa_dims,
            filename,
        )

def load_sim_and_soa_configs_from_dir(
    outdir: Path,
    sim_params_class: type[SoaSimulationParams],
    sim_params : dict[str, SoaSimulationParams],
    soa_conditions : dict[str, SoaConditions],
    soa_material : dict[str, SoaMaterial],
    soa_dims : dict[str, SoaDimensions],
    filename : str = 'config.toml',
):

    for case_dir in sorted(outdir.iterdir()):

        data_file = case_dir / filename

        if not case_dir.is_dir() or not data_file.is_file():
            continue

        sim_params[case_dir.name] = sim_params_class.from_toml(data_file)
        soa_material[case_dir.name] = SoaMaterial.from_toml(data_file)
        soa_dims[case_dir.name] = SoaDimensions.from_toml(data_file)
        soa_conditions[case_dir.name] = SoaConditions.from_toml(data_file)


def load_simulation_results_from_dir(
    outdir: Path,
    filename : str = 'soa_sim_results.npz',
) -> dict[str, dict[str, SoaSimulationResult]]:

    results = {}

    for case_dir in sorted(outdir.iterdir()):

        if not case_dir.is_dir():
            continue

        case_results = {}
        for run_dir in sorted(case_dir.iterdir()):

            data_file = run_dir / filename
            if not run_dir.is_dir() or not data_file.is_file():
                continue

            case_results[run_dir.name] = load_soa_sim_result(run_dir, filename=filename)

        if case_results:
            results[case_dir.name] = case_results

    return results


def get_simulation_processed_output_dirs(
    outdir: Path,
    filename : str = 'soa_sim_results.npz',
) -> dict[str, list[Path]]:

    results = {}

    for case_dir in sorted(outdir.iterdir()):

        if not case_dir.is_dir():
            continue

        case_results = []
        for run_dir in sorted(case_dir.iterdir()):

            data_file = run_dir / filename
            if not run_dir.is_dir() or not data_file.is_file():
                continue

            case_results.append(load_soa_sim_result(run_dir, filename=filename))

        if case_results:
            results[case_dir.name] = case_results

    return results


def add_args_to_parser(parser:ArgumentParser, sim_params_class: type[SoaSimulationParams]):

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
        default=None,
    )
    parser.add_argument("--config-file", type=Path, default="./sim_configs/SOA_simulation_Li_2008.toml")
    parser.add_argument("--soa-conditions-file", type=Path, default=None)
    parser.add_argument("--soa-material-file", type=Path, default=None)
    parser.add_argument("--soa-dims-file", type=Path, default=None)
    parser.add_argument("--sim-params-file", type=Path, default=None)

    for c in [sim_params_class, SoaMaterial, SoaDimensions, SoaConditions]:
        c.add_cli_args_to_parser(parser)

    parser.add_argument(
        "--sim-data-dir",
        type=Path,
        default=None,
        help="Directory containing previously generated simulation data. "
             "If present, simulation is skipped and the data stored in this dir is processed.",
    )

    parser.add_argument(
        "--no-processing",
        action='store_true',
        help="Skip the processing of the simulation data.",
    )

    
