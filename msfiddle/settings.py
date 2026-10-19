"""Load FIDDLE configuration files with defaults and consistency checks."""

from __future__ import annotations

import os
import warnings
from typing import Any

import yaml

POST_PROCESSING_DEFAULTS = {
    # 13C isotope offsets k to search: the precursor m/z may come from the
    # M+k isotope peak. [0] searches the measured mass only.
    "isotope_errors": [0],
    # Subtracted from a candidate's rescore per unit of k.
    "isotope_penalty": 0.5,
}


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a YAML config, fill post-processing defaults and validate it."""
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    post_processing = config.get("post_processing")
    if post_processing is not None:
        for key, value in POST_PROCESSING_DEFAULTS.items():
            post_processing.setdefault(key, value)
    validate_config(config, source=str(path))
    return config


def validate_config(config: dict[str, Any], source: str = "config") -> None:
    """Raise ValueError on invalid post-processing settings, and warn when a
    dataset filter admits precursor errors the refinement window cannot reach."""
    pp = config.get("post_processing")
    if pp is None:
        return

    errors = []
    if len(pp["refine_atom_type"]) != len(pp["refine_atom_num"]):
        errors.append("refine_atom_type and refine_atom_num differ in length")
    if pp["mass_tolerance"] <= 0:
        errors.append("mass_tolerance must be positive")
    if int(pp["top_k"]) < 1:
        errors.append("top_k must be at least 1")
    offsets = pp["isotope_errors"]
    if (
        not isinstance(offsets, list)
        or 0 not in offsets
        or len(set(offsets)) != len(offsets)
        or not all(isinstance(k, int) and k >= 0 for k in offsets)
    ):
        errors.append(
            "isotope_errors must be distinct non-negative integers including 0"
        )
    if pp["isotope_penalty"] < 0:
        errors.append("isotope_penalty must be non-negative")
    if errors:
        raise ValueError(f"{source}: " + "; ".join(errors))

    if not pp.get("ppm_mode", True):
        return  # mass_tolerance is in Da, not comparable with ppm filters
    for name, section in config.items():
        if (
            isinstance(section, dict)
            and "ppm_tolerance" in section
            and section["ppm_tolerance"] > pp["mass_tolerance"]
        ):
            warnings.warn(
                f"{source}: {name}.ppm_tolerance ({section['ppm_tolerance']}) is "
                f"wider than post_processing.mass_tolerance ({pp['mass_tolerance']}); "
                "spectra admitted beyond the refinement window cannot be "
                "predicted correctly",
                stacklevel=3,
            )
