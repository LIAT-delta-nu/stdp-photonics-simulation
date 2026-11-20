import tomllib
import tomli_w
from pathlib import Path
from dataclasses import fields
from typing import Self


def toml_config(section: str | None = None):
    def decorator(cls):
        @classmethod
        def from_toml(
            cls,
            path: str | Path,
            section: str | None = section,
        ) -> Self:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            if section is not None:
                subsections = section.replace(" ","").split(".")
                for subsection in subsections:
                    data = data[subsection]

            return cls(**data)

        def update_from_toml(self, path: str | Path, section: str=section) -> Self:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            if section is not None:
                subsections = section.replace(" ","").split(".")
                for subsection in subsections:
                    data = data[subsection]

            for key, value in data.items():
                if key in self.__dataclass_fields__:
                    setattr(self, key, value)

            return self

        def to_toml(self, path: str | Path, section: str=section):
            path = Path(path)
            if path.exists():
                with path.open("rb") as field:
                    data = tomllib.load(field)
            else:
                data = {}

            section_table = data.setdefault(section, {})

            for field in fields(self):
                if isinstance(getattr(self, field.name), (int, float, str, bool, list, tuple, dict)):
                    section_table[field.name]= getattr(self, field.name)

            with open(path, "wb") as f:
                tomli_w.dump(data, f)

        cls.update_from_toml = update_from_toml
        cls.from_toml = from_toml
        cls.to_toml = to_toml
        return cls

    return decorator