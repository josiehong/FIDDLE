"""
msfiddle: A package for predicting chemical formulas from tandem mass spectra
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("msfiddle")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

from .api import (
    MsFiddlePredictor,
    predict_batch_from_spectra,
    predict_from_mgf,
    predict_from_spectrum,
)
from .utils.mol_utils import vector_to_formula, formula_to_vector
from .utils.msms_utils import mass_calculator
from .utils.refine_utils import formula_refinement
from .download import check_models_exist, download_models, get_model_path


def __getattr__(name):
    """Lazy-load torch-dependent components only when accessed."""
    if name in ("MS2FNet_tcn", "FormulaEncoder", "RescoreHead"):
        from .model_tcn import FormulaEncoder, MS2FNet_tcn, RescoreHead

        globals()["MS2FNet_tcn"] = MS2FNet_tcn
        globals()["FormulaEncoder"] = FormulaEncoder
        globals()["RescoreHead"] = RescoreHead
        return globals()[name]
    if name in ("test_step", "rescore_candidates"):
        from .main import rescore_candidates, test_step

        globals()["test_step"] = test_step
        globals()["rescore_candidates"] = rescore_candidates
        return globals()[name]
    raise AttributeError(f"module 'msfiddle' has no attribute {name!r}")


__all__ = [
    "__version__",
    "MsFiddlePredictor",
    "predict_from_spectrum",
    "predict_batch_from_spectra",
    "predict_from_mgf",
    "MS2FNet_tcn",
    "FormulaEncoder",
    "RescoreHead",
    "formula_refinement",
    "mass_calculator",
    "vector_to_formula",
    "formula_to_vector",
    "test_step",
    "rescore_candidates",
    "download_models",
    "check_models_exist",
    "get_model_path",
]
