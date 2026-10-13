import tomllib
from pathlib import Path
from dataclasses import dataclass
from typing import Self


def toml_config(section: str | None = None):
    def decorator(cls):
        @classmethod
        def from_toml(
            cls,
            path: str | Path,
            section: str | None = section,
        ) -> Self:
            print(Path.cwd())
            with open(path, "rb") as f:
                data = tomllib.load(f)

            if section is not None:
                subsections = section.replace(" ","").split(".")
                for subsection in subsections:
                    data = data[subsection]

            return cls(**data)


        def update_from_toml(self, path: str | Path, section=section):
            with open(path, "rb") as f:
                data = tomllib.load(f)

            if section is not None:
                subsections = section.replace(" ","").split(".")
                for subsection in subsections:
                    data = data[subsection]

            for key, value in data.items():
                if key in self.__dataclass_fields__:
                    setattr(self, key, value)

        cls.update_from_toml = update_from_toml
        cls.from_toml = from_toml
        return cls

    return decorator