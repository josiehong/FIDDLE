"""Public prediction API for msfiddle."""

from __future__ import annotations

import math
import os
import re
import time
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyteomics import mgf

from .download import download_models as download_pretrained_models
from .download import get_checkpoint_dir
from .external_results import load_buddy_results, load_sirius_results
from .utils.mol_utils import (
    ATOMS_INDEX,
    formula_to_dict,
    formula_to_vector,
    vector_to_formula,
)
from .utils.msms_utils import mass_calculator
from .utils.pkl_utils import generate_ms, parse_collision_energy, unify_precursor_type
from .utils.refine_utils import formula_refinement

_INSTRUMENT_TYPES = {"orbitrap", "qtof"}
_NEUTRAL_FORMULAS = ("CH4O2", "CH2O2", "H2O", "NH3", "CO2")


@dataclass
class _EncodedSpectrum:
    id: str
    precursor_type: str
    precursor_mz: float
    collision_energy: str
    spec: np.ndarray
    env: np.ndarray
    neutral_add: np.ndarray


@dataclass
class _ModelPrediction:
    spectrum: _EncodedSpectrum
    vector_string: str
    pred_formula: str
    pred_mass: float
    pred_atom_count: float
    pred_hc_count: float
    prediction_time: float


class MsFiddlePredictor:
    """Reusable predictor for msfiddle formula inference.

    Instantiate this class once in long-running Python applications to avoid
    loading model checkpoints for each spectrum.
    """

    def __init__(
        self,
        *,
        instrument_type: str = "orbitrap",
        device: str | int | Sequence[int] | None = "cpu",
        no_cuda: bool = False,
        config_path: str | os.PathLike[str] | None = None,
        resume_path: str | os.PathLike[str] | None = None,
        rescore_resume_path: str | os.PathLike[str] | None = None,
        download_models: bool = False,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        self.instrument_type = _normalize_instrument_type(instrument_type)
        self.requested_device = device
        self.no_cuda = no_cuda
        self.download_models = download_models
        self.batch_size = _validate_batch_size(batch_size)
        self.verbose = verbose

        package_dir = Path(__file__).resolve().parent
        self.config_path = (
            Path(config_path)
            if config_path
            else (package_dir / "config" / f"fiddle_tcn_{self.instrument_type}.yml")
        )
        self.config = _load_config(self.config_path)

        checkpoint_dir = Path(get_checkpoint_dir())
        self.resume_path = (
            Path(resume_path)
            if resume_path
            else (checkpoint_dir / f"fiddle_tcn_{self.instrument_type}.pt")
        )
        self.rescore_resume_path = (
            Path(rescore_resume_path)
            if rescore_resume_path
            else checkpoint_dir / f"fiddle_rescore_{self.instrument_type}.pt"
        )

        self._torch = None
        self._nn = None
        self._F = None
        self.device = None
        self.device_ids: list[int] = []
        self.model = None
        self.formula_encoder = None
        self.rescore_head = None
        self._models_loaded = False

    def predict_spectrum(
        self,
        mz_array: Sequence[float],
        intensity_array: Sequence[float],
        precursor_mz: float,
        adduct: str,
        *,
        top_k: int = 5,
        collision_energy: str | float = "Unknown",
    ) -> list[dict[str, Any]]:
        """Predict formula candidates for one native MS/MS spectrum."""

        result = self.predict_batch(
            [
                {
                    "id": "spectrum_0",
                    "mz_array": mz_array,
                    "intensity_array": intensity_array,
                    "precursor_mz": precursor_mz,
                    "adduct": adduct,
                    "collision_energy": collision_energy,
                }
            ],
            top_k=top_k,
        )
        return result[0]["candidates"]

    def predict_batch(
        self,
        spectra: Sequence[Mapping[str, Any]],
        *,
        top_k: int = 5,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Predict formula candidates for multiple native MS/MS spectra.

        Each spectrum mapping must contain ``mz_array``, ``intensity_array``,
        ``precursor_mz``, and ``adduct``. Optional keys are ``collision_energy``
        and ``id``.
        """

        top_k = _validate_top_k(top_k)
        batch_size = (
            self.batch_size if batch_size is None else _validate_batch_size(batch_size)
        )
        encoded = [
            self._encode_spectrum_record(spectrum, idx)
            for idx, spectrum in enumerate(_validate_spectra_collection(spectra))
        ]
        predictions = self._predict_encoded(encoded, batch_size=batch_size)

        records = []
        for prediction in predictions:
            start_time = time.time()
            refined_results, neutral_mass = self._refine_and_rescore(prediction, top_k)
            refinement_time = time.time() - start_time
            metadata = self._metadata_for_prediction(
                prediction,
                neutral_mass,
                running_time=prediction.prediction_time + refinement_time,
            )
            records.append(
                {
                    "id": prediction.spectrum.id,
                    "candidates": self._format_candidates(
                        refined_results,
                        prediction,
                        neutral_mass,
                    ),
                    "metadata": metadata,
                }
            )
        return records

    def predict_mgf(
        self,
        test_data_path: str | os.PathLike[str],
        *,
        buddy_path: str | os.PathLike[str] | None = "",
        sirius_path: str | os.PathLike[str] | None = "",
        top_k: int | None = None,
        batch_size: int | None = None,
    ) -> pd.DataFrame:
        """Predict formulas for spectra in an MGF file.

        ``buddy_path`` and ``sirius_path`` may point to either native/original
        BUDDY/msbuddy and SIRIUS formula-identification outputs or legacy
        msfiddle-normalized CSV files.

        The returned DataFrame uses the same result columns as the CLI CSV.
        """

        top_k = (
            _validate_top_k(top_k)
            if top_k is not None
            else int(self.config["post_processing"]["top_k"])
        )
        batch_size = (
            self.batch_size if batch_size is None else _validate_batch_size(batch_size)
        )
        encoded = self._load_mgf_spectra(test_data_path)
        predictions = self._predict_encoded(encoded, batch_size=batch_size)

        buddy_df = load_buddy_results(buddy_path) if buddy_path else None
        sirius_df = load_sirius_results(sirius_path) if sirius_path else None

        refined_formula = {f"Refined Formula ({k})": [] for k in range(top_k)}
        refined_mass = {f"Refined Mass ({k})": [] for k in range(top_k)}
        refined_rescore = {f"Rescore ({k})": [] for k in range(top_k)}
        running_time: list[float] = []
        neutral_masses: list[float] = []

        prediction_iter = predictions
        if self.verbose and predictions:
            from tqdm import tqdm

            prediction_iter = tqdm(predictions, total=len(predictions), desc="Post")

        for prediction in prediction_iter:
            start_time = time.time()
            extra_formulas = self._external_formula_candidates(
                prediction.spectrum.id,
                buddy_df=buddy_df,
                sirius_df=sirius_df,
            )
            results, neutral_mass = self._refine_and_rescore(
                prediction,
                top_k,
                extra_formulas=extra_formulas,
            )
            neutral_masses.append(neutral_mass)
            running_time.append(prediction.prediction_time + time.time() - start_time)

            for idx, (formula, mass, score) in enumerate(
                zip(results["formula"], results["mass"], results["rescore"])
            ):
                refined_formula[f"Refined Formula ({idx})"].append(formula)
                refined_mass[f"Refined Mass ({idx})"].append(mass)
                refined_rescore[f"Rescore ({idx})"].append(score)

        base_columns = {
            "ID": [prediction.spectrum.id for prediction in predictions],
            "Y Pred": [prediction.vector_string for prediction in predictions],
            "Mass": neutral_masses,
            "Pred Formula": [prediction.pred_formula for prediction in predictions],
            "Pred Mass": [prediction.pred_mass for prediction in predictions],
            "Pred Atom Num": [prediction.pred_atom_count for prediction in predictions],
            "Pred H/C Num": [prediction.pred_hc_count for prediction in predictions],
            "Running Time": running_time,
        }
        return pd.DataFrame(
            {**base_columns, **refined_formula, **refined_mass, **refined_rescore}
        )

    def _encode_spectrum_record(
        self,
        spectrum: Mapping[str, Any],
        idx: int,
    ) -> _EncodedSpectrum:
        for key in ("mz_array", "intensity_array", "precursor_mz", "adduct"):
            if key not in spectrum:
                raise ValueError(f"Malformed spectrum at index {idx}: missing '{key}'")
        return self._encode_spectrum(
            mz_array=spectrum["mz_array"],
            intensity_array=spectrum["intensity_array"],
            precursor_mz=spectrum["precursor_mz"],
            adduct=spectrum["adduct"],
            collision_energy=spectrum.get("collision_energy", "Unknown"),
            spec_id=str(spectrum.get("id", f"spectrum_{idx}")),
        )

    def _encode_spectrum(
        self,
        *,
        mz_array: Sequence[float],
        intensity_array: Sequence[float],
        precursor_mz: float,
        adduct: str,
        collision_energy: str | float,
        spec_id: str,
    ) -> _EncodedSpectrum:
        encoder = self.config["encoding"]
        mz_values, intensity_values = _validate_arrays(
            mz_array,
            intensity_array,
            min_peak_num=5,
            max_mz=float(encoder["max_mz"]),
            min_mz=50.0,
        )
        precursor_mz = _validate_float("precursor_mz", precursor_mz)
        precursor_type = _validate_adduct(adduct, encoder)
        charge = int(encoder["type2charge"][unify_precursor_type(precursor_type)])

        good_spec, _, _, spec_arr = generate_ms(
            x=mz_values,
            y=intensity_values,
            precursor_mz=precursor_mz,
            resolution=encoder["resolution"],
            max_mz=encoder["max_mz"],
            charge=charge,
        )
        if not good_spec:
            raise ValueError("Invalid spectrum: unable to encode mass spectrum")

        neutral_add, adjusted_precursor_type = _melt_neutral_precursor(precursor_type)
        if adjusted_precursor_type not in encoder["precursor_type"]:
            raise ValueError(
                "Unsupported adduct after neutral-loss normalization: "
                f"{adjusted_precursor_type}"
            )

        ce_str = _collision_energy_to_string(collision_energy)
        ce, nce = parse_collision_energy(
            ce_str=ce_str,
            precursor_mz=precursor_mz,
            charge=abs(charge),
        )
        if ce is None and nce is None:
            raise ValueError(f"Could not parse collision energy: {collision_energy!r}")

        env_arr = np.array(
            [
                precursor_mz,
                nce,
                encoder["precursor_type"][adjusted_precursor_type],
            ],
            dtype=np.float32,
        )
        return _EncodedSpectrum(
            id=spec_id,
            precursor_type=precursor_type,
            precursor_mz=precursor_mz,
            collision_energy=ce_str,
            spec=spec_arr[:, 0].astype(np.float32),
            env=env_arr,
            neutral_add=np.array(neutral_add, dtype=np.float32),
        )

    def _load_mgf_spectra(
        self, test_data_path: str | os.PathLike[str]
    ) -> list[_EncodedSpectrum]:
        path = Path(test_data_path)
        if not path.exists():
            raise FileNotFoundError(f"MGF file not found: {path}")

        spectra = list(mgf.read(str(path)))
        if self.verbose:
            print(len(spectra), "spectra loaded from", str(path))

        precursor_mz_key = (
            "simulated_precursor_mz"
            if self.config["encoding"]["use_simulated_precursor_mz"]
            else "precursor_mz"
        )
        encoded: list[_EncodedSpectrum] = []
        invalid_count = 0
        for spectrum in spectra:
            params = spectrum.get("params", {})
            missing = [
                key
                for key in (
                    "title",
                    precursor_mz_key,
                    "precursor_type",
                    "collision_energy",
                )
                if key not in params
            ]
            if missing:
                invalid_count += 1
                if self.verbose:
                    print(
                        "MGFError: lacking necessary keys in mgf file, skip this spectrum"
                    )
                    print(
                        "expected keys in mgf: "
                        f"('title', '{precursor_mz_key}', 'precursor_type', 'collision_energy')"
                    )
                continue
            try:
                encoded.append(
                    self._encode_spectrum(
                        mz_array=spectrum["m/z array"],
                        intensity_array=spectrum["intensity array"],
                        precursor_mz=float(params[precursor_mz_key]),
                        adduct=params["precursor_type"],
                        collision_energy=params["collision_energy"],
                        spec_id=str(params["title"]),
                    )
                )
            except ValueError:
                invalid_count += 1
                continue

        if self.verbose:
            print(len(spectra) - invalid_count, "spectra left after filtering")
            print(len(encoded), "spectra loaded into the dataset")
        return encoded

    def _predict_encoded(
        self,
        encoded: Sequence[_EncodedSpectrum],
        *,
        batch_size: int,
    ) -> list[_ModelPrediction]:
        if not encoded:
            return []
        self._ensure_models_loaded()
        torch = self._torch
        assert torch is not None

        start_time = time.time()
        formula_vectors: list[np.ndarray] = []
        mass_predictions: list[np.ndarray] = []
        atom_predictions: list[np.ndarray] = []
        hc_predictions: list[np.ndarray] = []

        batch_starts: Sequence[int] = range(0, len(encoded), batch_size)
        if self.verbose:
            from tqdm import tqdm

            batch_starts = tqdm(
                batch_starts, total=math.ceil(len(encoded) / batch_size), desc="Eval"
            )

        assert self.model is not None
        self.model.eval()
        for start in batch_starts:
            batch = encoded[start : start + batch_size]
            spec_t = torch.from_numpy(np.stack([item.spec for item in batch])).to(
                self.device,
                dtype=torch.float32,
            )
            env_t = torch.from_numpy(np.stack([item.env for item in batch])).to(
                self.device,
                dtype=torch.float32,
            )
            neutral_t = torch.from_numpy(
                np.stack([item.neutral_add for item in batch])
            ).to(
                self.device,
                dtype=torch.float32,
            )
            with torch.no_grad():
                _, pred_f, pred_mass, pred_atomnum, pred_hcnum = self.model(
                    spec_t, env_t
                )
            pred_f = pred_f - neutral_t
            formula_vectors.append(pred_f.detach().cpu().numpy())
            mass_predictions.append(pred_mass.detach().cpu().numpy())
            atom_predictions.append(pred_atomnum.detach().cpu().numpy())
            hc_predictions.append(pred_hcnum.detach().cpu().numpy())

        prediction_time = (time.time() - start_time) / len(encoded)
        formula_arr = np.concatenate(formula_vectors, axis=0)
        mass_arr = np.concatenate(mass_predictions, axis=0)
        atom_arr = np.concatenate(atom_predictions, axis=0)
        hc_arr = np.concatenate(hc_predictions, axis=0)

        predictions = []
        for idx, item in enumerate(encoded):
            vector = formula_arr[idx]
            predictions.append(
                _ModelPrediction(
                    spectrum=item,
                    vector_string=";".join(vector.astype("str")),
                    pred_formula=vector_to_formula(vector),
                    pred_mass=float(mass_arr[idx]),
                    pred_atom_count=float(atom_arr[idx]),
                    pred_hc_count=float(hc_arr[idx]),
                    prediction_time=prediction_time,
                )
            )
        return predictions

    def _ensure_models_loaded(self) -> None:
        if self._models_loaded:
            return
        self._ensure_model_files()
        self._load_torch_dependencies()
        self._load_models()
        self._models_loaded = True

    def _ensure_model_files(self) -> None:
        missing_paths = [
            path
            for path in (self.resume_path, self.rescore_resume_path)
            if not path.exists()
        ]
        if missing_paths and self.download_models:
            download_pretrained_models(
                models=[
                    f"fiddle_tcn_{self.instrument_type}",
                    f"fiddle_rescore_{self.instrument_type}",
                ]
            )
            missing_paths = [
                path
                for path in (self.resume_path, self.rescore_resume_path)
                if not path.exists()
            ]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(
                "Missing msfiddle checkpoint file(s): "
                f"{missing}. Run 'msfiddle-download-models' or pass "
                "download_models=True."
            )

    def _load_torch_dependencies(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError(
                "msfiddle inference requires PyTorch. Install it with "
                "'pip install \"msfiddle[inference]\"' or follow the PyTorch "
                "installation guide for your platform."
            ) from exc
        self._torch = torch
        self._nn = nn
        self._F = F

    def _load_models(self) -> None:
        torch = self._torch
        nn = self._nn
        assert torch is not None
        assert nn is not None

        from .model_tcn import FormulaEncoder, MS2FNet_tcn, RescoreHead

        self.device, self.device_ids = _resolve_torch_device(
            torch,
            self.requested_device,
            self.no_cuda,
        )

        if self.verbose:
            print(f"Loaded model & training configuration from {self.config_path}")
            print(f"Device(s): {self.requested_device}")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.nn\.utils\.weight_norm` is deprecated",
                category=FutureWarning,
            )
            model = MS2FNet_tcn(self.config["model"]).to(self.device)
        if self.verbose:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"# MS2FNet_tcn Params: {num_params}")
            print("Loading the formula prediction model...")
        state_dict = torch.load(
            self.resume_path,
            map_location=self.device,
            weights_only=False,
        )["model_state_dict"]
        model.load_state_dict(_strip_module_prefix(state_dict))
        model.eval()
        if len(self.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.device_ids)

        formula_encoder = FormulaEncoder(self.config["model"]).to(self.device)
        rescore_head = RescoreHead(self.config["model"]).to(self.device)
        if self.verbose:
            n_params = sum(p.numel() for p in formula_encoder.parameters()) + sum(
                p.numel() for p in rescore_head.parameters()
            )
            print(f"# Rescore Params: {n_params}")
            print("Loading the rescore model...")
        ckpt = torch.load(
            self.rescore_resume_path,
            map_location=self.device,
            weights_only=False,
        )
        formula_encoder.load_state_dict(ckpt["formula_encoder_state_dict"])
        rescore_head.load_state_dict(ckpt["rescore_head_state_dict"])
        formula_encoder.eval()
        rescore_head.eval()

        self.model = model
        self.formula_encoder = formula_encoder
        self.rescore_head = rescore_head

    def _refine_and_rescore(
        self,
        prediction: _ModelPrediction,
        top_k: int,
        *,
        extra_formulas: Sequence[str] | None = None,
    ) -> tuple[dict[str, list[Any]], float]:
        neutral_mass = float(
            mass_calculator(
                prediction.spectrum.precursor_type,
                prediction.spectrum.precursor_mz,
            )
        )
        f0_list = [prediction.pred_formula]
        if extra_formulas:
            f0_list.extend(extra_formulas)
        f0_list = _dedupe_nonempty(f0_list)

        refine_atom_type = list(self.config["post_processing"]["refine_atom_type"])
        refine_atom_num = list(self.config["post_processing"]["refine_atom_num"])
        for formula in f0_list:
            for atom, count in formula_to_dict(formula).items():
                if atom == "H" or atom in refine_atom_type:
                    continue
                refine_atom_type.append(atom)
                refine_atom_num.append(max(1, int(count)))

        refined_results = formula_refinement(
            f0_list,
            neutral_mass,
            self.config["post_processing"]["mass_tolerance"],
            self.config["post_processing"]["ppm_mode"],
            top_k,
            self.config["post_processing"]["maxium_miss_atom_num"],
            self.config["post_processing"]["time_out"],
            refine_atom_type,
            refine_atom_num,
        )
        return (
            self._rescore_candidates(
                prediction.spectrum,
                refined_results,
                top_k,
            ),
            neutral_mass,
        )

    def _rescore_candidates(
        self,
        spectrum: _EncodedSpectrum,
        refined_results: dict[str, list[Any]],
        top_k: int,
    ) -> dict[str, list[Any]]:
        torch = self._torch
        F = self._F
        assert torch is not None
        assert F is not None
        assert self.model is not None
        assert self.formula_encoder is not None
        assert self.rescore_head is not None

        refine_f = [
            formula for formula in refined_results["formula"] if formula is not None
        ]
        refine_m = [
            mass
            for formula, mass in zip(
                refined_results["formula"], refined_results["mass"]
            )
            if formula is not None
        ]
        if not refine_f:
            refined_results["rescore"] = [0.0] * top_k
            return refined_results

        f_vecs = torch.from_numpy(
            np.array([formula_to_vector(formula) for formula in refine_f])
        )
        spec_t = torch.from_numpy(spectrum.spec[None, :]).to(
            self.device, dtype=torch.float32
        )
        env_t = (
            torch.from_numpy(spectrum.env[None, :])
            .to(self.device, dtype=torch.float32)
            .clone()
        )
        env_t[:, 0] = 0.0

        self.model.eval()
        self.formula_encoder.eval()
        self.rescore_head.eval()
        with torch.no_grad():
            z_spec, _, _, _, _ = self.model(spec_t, env_t)
            z_spec = F.normalize(z_spec, dim=1)
            z_spec_rep = z_spec.expand(len(refine_f), -1)
            f_t = f_vecs.to(self.device, dtype=torch.float32)
            z_form = self.formula_encoder(f_t)
            logits = self.rescore_head(z_spec_rep * z_form)
            scores = torch.sigmoid(logits).detach().cpu().numpy()

        ranked = sorted(
            zip(scores, refine_f, refine_m), key=lambda item: item[0], reverse=True
        )
        sorted_rescore, sorted_f, sorted_m = map(list, zip(*ranked))
        while len(sorted_f) < top_k:
            sorted_f.append(None)
            sorted_m.append(None)
            sorted_rescore.append(0.0)
        return {"formula": sorted_f, "mass": sorted_m, "rescore": sorted_rescore}

    def _format_candidates(
        self,
        refined_results: dict[str, list[Any]],
        prediction: _ModelPrediction,
        neutral_mass: float,
    ) -> list[dict[str, Any]]:
        candidates = []
        base_metadata = self._metadata_for_prediction(prediction, neutral_mass)
        for formula, mass, raw_score in zip(
            refined_results["formula"],
            refined_results["mass"],
            refined_results["rescore"],
        ):
            if formula is None:
                continue
            metadata = dict(base_metadata)
            metadata["raw_rescore"] = float(raw_score)
            candidates.append(
                {
                    "formula": formula,
                    "score": _normalize_score(raw_score),
                    "mass": None if mass is None else float(mass),
                    "metadata": metadata,
                }
            )
        return candidates

    def _metadata_for_prediction(
        self,
        prediction: _ModelPrediction,
        neutral_mass: float,
        *,
        running_time: float | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "pred_formula": prediction.pred_formula,
            "pred_mass": prediction.pred_mass,
            "pred_atom_count": prediction.pred_atom_count,
            "pred_hc_count": prediction.pred_hc_count,
            "neutral_mass": neutral_mass,
            "instrument_type": self.instrument_type,
            "collision_energy": prediction.spectrum.collision_energy,
            "adduct": prediction.spectrum.precursor_type,
        }
        if running_time is not None:
            metadata["running_time"] = running_time
        return metadata

    def _external_formula_candidates(
        self,
        spec_id: str,
        *,
        buddy_df: pd.DataFrame | None,
        sirius_df: pd.DataFrame | None,
    ) -> list[str]:
        candidates: list[str] = []
        if buddy_df is not None:
            rows = buddy_df.loc[buddy_df["ID"] == spec_id]
            if len(rows) > 0:
                row = rows.iloc[0]
                formulas = row[
                    [
                        "Pred Formula (1)",
                        "Pred Formula (2)",
                        "Pred Formula (3)",
                        "Pred Formula (4)",
                        "Pred Formula (5)",
                    ]
                ].tolist()
                scores = row[
                    [
                        "BUDDY Score (1)",
                        "BUDDY Score (2)",
                        "BUDDY Score (3)",
                        "BUDDY Score (4)",
                        "BUDDY Score (5)",
                    ]
                ].tolist()
                candidates.extend(
                    formula
                    for formula, score in zip(formulas, scores)
                    if _is_external_formula(formula)
                    and pd.notna(score)
                    and score < self.config["post_processing"]["buddy_fdr_thr"]
                )
        if sirius_df is not None:
            rows = sirius_df.loc[sirius_df["ID"] == spec_id]
            if len(rows) > 0:
                row = rows.iloc[0]
                formulas = row[
                    [
                        "Pred Formula (1)",
                        "Pred Formula (2)",
                        "Pred Formula (3)",
                        "Pred Formula (4)",
                        "Pred Formula (5)",
                    ]
                ].tolist()
                scores = row[
                    [
                        "SIRIUS Score (1)",
                        "SIRIUS Score (2)",
                        "SIRIUS Score (3)",
                        "SIRIUS Score (4)",
                        "SIRIUS Score (5)",
                    ]
                ].tolist()
                candidates.extend(
                    formula
                    for formula, score in zip(formulas, scores)
                    if _is_external_formula(formula)
                    and pd.notna(score)
                    and score > self.config["post_processing"]["sirius_score_thr"]
                )
        return candidates


def predict_from_spectrum(
    mz_array: Sequence[float],
    intensity_array: Sequence[float],
    precursor_mz: float,
    adduct: str,
    *,
    top_k: int = 5,
    instrument_type: str = "orbitrap",
    collision_energy: str | float = "Unknown",
    device: str | int | Sequence[int] | None = "cpu",
    no_cuda: bool = False,
    config_path: str | os.PathLike[str] | None = None,
    resume_path: str | os.PathLike[str] | None = None,
    rescore_resume_path: str | os.PathLike[str] | None = None,
    download_models: bool = False,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Predict formula candidates for one native MS/MS spectrum."""

    predictor = MsFiddlePredictor(
        instrument_type=instrument_type,
        device=device,
        no_cuda=no_cuda,
        config_path=config_path,
        resume_path=resume_path,
        rescore_resume_path=rescore_resume_path,
        download_models=download_models,
        batch_size=batch_size,
    )
    return predictor.predict_spectrum(
        mz_array,
        intensity_array,
        precursor_mz,
        adduct,
        top_k=top_k,
        collision_energy=collision_energy,
    )


def predict_batch_from_spectra(
    spectra: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
    instrument_type: str = "orbitrap",
    device: str | int | Sequence[int] | None = "cpu",
    no_cuda: bool = False,
    config_path: str | os.PathLike[str] | None = None,
    resume_path: str | os.PathLike[str] | None = None,
    rescore_resume_path: str | os.PathLike[str] | None = None,
    download_models: bool = False,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Predict formula candidates for multiple native MS/MS spectra."""

    predictor = MsFiddlePredictor(
        instrument_type=instrument_type,
        device=device,
        no_cuda=no_cuda,
        config_path=config_path,
        resume_path=resume_path,
        rescore_resume_path=rescore_resume_path,
        download_models=download_models,
        batch_size=batch_size,
    )
    return predictor.predict_batch(spectra, top_k=top_k, batch_size=batch_size)


def predict_from_mgf(
    test_data_path: str | os.PathLike[str],
    *,
    instrument_type: str = "orbitrap",
    buddy_path: str | os.PathLike[str] | None = "",
    sirius_path: str | os.PathLike[str] | None = "",
    device: str | int | Sequence[int] | None = "cpu",
    no_cuda: bool = False,
    config_path: str | os.PathLike[str] | None = None,
    resume_path: str | os.PathLike[str] | None = None,
    rescore_resume_path: str | os.PathLike[str] | None = None,
    download_models: bool = False,
    top_k: int | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Predict formulas for spectra in an MGF file."""

    predictor = MsFiddlePredictor(
        instrument_type=instrument_type,
        device=device,
        no_cuda=no_cuda,
        config_path=config_path,
        resume_path=resume_path,
        rescore_resume_path=rescore_resume_path,
        download_models=download_models,
        batch_size=batch_size,
    )
    return predictor.predict_mgf(
        test_data_path,
        buddy_path=buddy_path,
        sirius_path=sirius_path,
        top_k=top_k,
        batch_size=batch_size,
    )


def _normalize_instrument_type(instrument_type: str) -> str:
    normalized = str(instrument_type).lower()
    if normalized not in _INSTRUMENT_TYPES:
        raise ValueError(
            "instrument_type must be one of "
            f"{sorted(_INSTRUMENT_TYPES)}, got {instrument_type!r}"
        )
    return normalized


def _is_external_formula(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "na", "nan", "none", "null", "<na>"}


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def _validate_top_k(top_k: int) -> int:
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    return top_k


def _validate_batch_size(batch_size: int) -> int:
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    return batch_size


def _validate_spectra_collection(
    spectra: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    if isinstance(spectra, (str, bytes)) or not isinstance(spectra, Sequence):
        raise ValueError("spectra must be a sequence of spectrum mappings")
    for idx, spectrum in enumerate(spectra):
        if not isinstance(spectrum, Mapping):
            raise ValueError(f"Malformed spectrum at index {idx}: expected a mapping")
    return spectra


def _validate_arrays(
    mz_array: Sequence[float],
    intensity_array: Sequence[float],
    *,
    min_peak_num: int,
    min_mz: float,
    max_mz: float,
) -> tuple[np.ndarray, np.ndarray]:
    mz_values = np.asarray(mz_array, dtype=float)
    intensity_values = np.asarray(intensity_array, dtype=float)
    if mz_values.ndim != 1 or intensity_values.ndim != 1:
        raise ValueError("mz_array and intensity_array must be one-dimensional")
    if len(mz_values) != len(intensity_values):
        raise ValueError(
            "mz_array and intensity_array must have the same length: "
            f"{len(mz_values)} != {len(intensity_values)}"
        )
    if len(mz_values) < min_peak_num:
        raise ValueError(f"Spectrum must contain at least {min_peak_num} peaks")
    if not np.isfinite(mz_values).all() or not np.isfinite(intensity_values).all():
        raise ValueError("mz_array and intensity_array must contain only finite values")
    if not np.isfinite(intensity_values.max() - intensity_values.min()):
        raise ValueError("Invalid intensity array")
    if float(np.max(mz_values)) < min_mz:
        raise ValueError(f"Spectrum maximum m/z must be at least {min_mz}")
    if float(np.max(mz_values)) > max_mz:
        raise ValueError(f"Spectrum maximum m/z must be no greater than {max_mz}")
    return mz_values, intensity_values


def _validate_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_adduct(adduct: Any, encoder: Mapping[str, Any]) -> str:
    precursor_type = str(adduct)
    normalized = unify_precursor_type(precursor_type)
    if normalized not in encoder["type2charge"]:
        raise ValueError(f"Unsupported adduct: {adduct!r}")
    return precursor_type


def _collision_energy_to_string(collision_energy: str | float) -> str:
    if isinstance(collision_energy, (int, float)):
        return str(float(collision_energy))
    return str(collision_energy)


def _melt_neutral_precursor(precursor_type: str) -> tuple[list[int], str]:
    precursor_type = unify_precursor_type(precursor_type)
    neutral_counts = {}
    for neutral in _NEUTRAL_FORMULAS:
        if neutral in precursor_type:
            pattern = r"([-+]?\d*)" + neutral
            count = re.findall(pattern, precursor_type)
            if count:
                value: str | int = count[0]
                if value == "+":
                    value = 1
                elif value == "-":
                    value = -1
                else:
                    value = int(value)
                neutral_counts[neutral] = value
                break

    pattern = r"([-+]?\d*)(?:" + "|".join(_NEUTRAL_FORMULAS) + ")"
    adjusted_precursor_type = re.sub(pattern, "", precursor_type)

    neutral_add = {}
    for neutral, count in neutral_counts.items():
        for atom, value in formula_to_dict(neutral).items():
            neutral_add[atom] = neutral_add.get(atom, 0) + value * count

    vector = [0] * len(ATOMS_INDEX)
    for atom, count in neutral_add.items():
        index = ATOMS_INDEX.get(atom)
        if index is not None:
            vector[index] = int(count) if count else 1
    return vector, adjusted_precursor_type


def _resolve_torch_device(
    torch: Any,
    device: str | int | Sequence[int] | None,
    no_cuda: bool,
) -> tuple[Any, list[int]]:
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu"), []
    if device is None:
        return torch.device("cpu"), []
    if isinstance(device, str):
        if device.lower() == "cpu":
            return torch.device("cpu"), []
        if device.startswith("cuda"):
            if ":" in device:
                idx = int(device.split(":", 1)[1])
            else:
                idx = 0
            return torch.device(f"cuda:{idx}"), [idx]
        idx = int(device)
        return torch.device(f"cuda:{idx}"), [idx]
    if isinstance(device, int):
        return torch.device(f"cuda:{device}"), [device]
    device_ids = [int(item) for item in device]
    if not device_ids:
        return torch.device("cpu"), []
    return torch.device(f"cuda:{device_ids[0]}"), device_ids


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    stripped = OrderedDict()
    for key, value in state_dict.items():
        stripped[key[7:] if key.startswith("module.") else key] = value
    return stripped


def _dedupe_nonempty(values: Sequence[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value or str(value) == "nan":
            continue
        value = str(value)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_score(score: Any) -> float:
    score = float(score)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
