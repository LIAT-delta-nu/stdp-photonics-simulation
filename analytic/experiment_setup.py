
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import copy

from devices import (
    SoaConditions,
    SoaMaterial,
    SoaDimensions,
    SoaSimulationParams
)

@dataclass
class ExperimentSetup:

    run_name: str
    timestamp: str
    sim_type_name: str

    cases: list[str]
    default_case: str

    soa_conditions: dict[str, SoaConditions]
    soa_material: dict[str, SoaMaterial]
    soa_dims: dict[str, SoaDimensions]
    sim_params: dict[str, SoaSimulationParams]

    @property
    def full_run_name(self):
        return (
            f"{self.timestamp}__{self.sim_type_name}__{self.run_name}"
            if self.run_name
            else f"{self.timestamp}__{self.sim_type_name}"
        )

    @classmethod
    def from_args(
        cls,
        args,
        sim_params_class: type[SoaSimulationParams],
        cases_if_missing: list[str],
    ) -> ExperimentSetup:

        # Run name
        run_name = args.run_name or input(
            "Enter run name and press enter "
            "(if no run name is wanted just press enter):\n"
        )

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return cls(
            run_name=run_name,
            timestamp=timestamp,
            sim_type_name=sim_params["default"].sim_type_name,
            cases=cases,
            default_case=default_case,
            soa_conditions=soa_conditions,
            soa_material=soa_material,
            soa_dims=soa_dims,
            sim_params=sim_params,
        )


def build_parser(sim_params_class: type[SoaSimulationParams]):
    parser = ArgumentParser()

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

    return parser
