import argparse
import numpy as np

from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import get_type_hints, get_origin, get_args, Union
import types


def cli_config(cls):
    if not is_dataclass(cls):
        raise TypeError("cli_dataclass must decorate a dataclass")

    @classmethod
    def add_cli_args_to_parser(cls, parser: argparse.ArgumentParser):

        hints = get_type_hints(cls)

        for field in fields(cls):
            annotation = hints[field.name]
            # Handle unions, e.g. float | np.ndarray
            args = get_args(annotation)
            types_to_check = args if args else (annotation,)

            if int in types_to_check:
                arg_type = int

            elif float in types_to_check:
                arg_type = float

            elif (
                np.ndarray in types_to_check
                or dict in types_to_check
                or list in types_to_check
            ):
                arg_type = Path

            else:
                continue

            parser.add_argument(
                f"--{field.name.replace('_', '-')}",
                type=arg_type,
                default=argparse.SUPPRESS,
                required=False,
            )

    def update_from_cli_args(self, args: argparse.Namespace) -> Self:
        for field in fields(self):
            if hasattr(args, field.name):
                setattr(self, field.name, getattr(args, field.name))
        return self

    cls.add_cli_args_to_parser = add_cli_args_to_parser
    cls.update_from_cli_args = update_from_cli_args

    return cls