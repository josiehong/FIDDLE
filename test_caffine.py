#!/usr/bin/env python3
"""
Test FIDDLE formula prediction on caffeine (C8H10N4O2) spectra from multiple sources.

Sources tested
--------------
- GNPS CCMSLIB00016149314 (fetched live via USI)
- NIST20 / NIST23 orbitrap spectra at NCE = 40 / 50 / 75 %
- MoNA orbitrap spectrum (HCD NCE 40 %)

Key findings
------------
1. GNPS caffeine spectra are excluded from training.
   All GNPS caffeine entries have SOURCE_INSTRUMENT=ftms (how Thermo instruments
   self-report in raw files), while the gnps_orbitrap config requires an exact
   match to 'orbitrap'. Since FTMS and Orbitrap are the same instrument family,
   adding 'ftms' to the instrument allowlist in the config would fix this.

2. The FDR model consistently prefers C10H12NO3 over caffeine.
   Even on in-training NIST spectra, caffeine (ppm ~2.5) scores lower than
   C10H12NO3 (ppm ~9.5) except at NCE=75% where more fragment peaks are present.
   The FDR model appears to have a systematic bias toward C10H12NO3 near this mass.

3. Atom type extension in refinement is critical.
   run_fiddle.py (and prepare_fdr.py) pass refine_atom_type = ['C','O','N','H']
   directly from config without extending it with atoms from the initial prediction.
   When the model predicts a rare atom (e.g. F, S), that atom is frozen in the
   refinement search — formulas without it (like caffeine) become completely
   unreachable. Adding the atom extension logic to both run_fiddle.py and
   prepare_fdr.py (and retraining the FDR model) would fix this.

4. prepare_fdr.py has a misleading comment (line ~185).
   The comment says "experimental precursor m/z" but the code actually uses
   mass_true (the exact theoretical monoisotopic mass from the true formula).

5. Score combination (FDR × exp(−ppm / λ)) partially helps.
   With λ=5, caffeine reaches rank 1 for NIST NCE=40% spectra but not for the
   GNPS spectrum (FDR gap is 18× there). The real fix is retraining the FDR model
   with corrected training data (items 1 and 3 above).
"""

import os
import sys
import tempfile
import time
from collections import OrderedDict

import numpy as np
import requests
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup — run from the FIDDLE repo root
# ---------------------------------------------------------------------------
FIDDLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIDDLE_DIR)

from dataset import MGFDataset
from model_tcn import MS2FNet_tcn, FDRNet
from utils import (
    formula_refinement,
    mass_calculator,
    vector_to_formula,
    formula_to_vector,
)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(FIDDLE_DIR, "config", "fiddle_tcn_orbitrap.yml")
MODEL_PATH = os.path.join(FIDDLE_DIR, "check_point", "fiddle_tcn_orbitrap.pt")
FDR_MODEL_PATH = os.path.join(FIDDLE_DIR, "check_point", "fiddle_fdr_orbitrap.pt")

# ---------------------------------------------------------------------------
# Load config + models (once)
# ---------------------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)


DEVICE = torch.device("cpu")


def _load_state(path):
    sd = torch.load(path, map_location=DEVICE, weights_only=False)["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = OrderedDict((k.removeprefix("module."), v) for k, v in sd.items())
    return sd


print("Loading FIDDLE models …")
model = MS2FNet_tcn(CONFIG["model"]).to(DEVICE)
model.load_state_dict(_load_state(MODEL_PATH))
model.eval()

fdr_model = FDRNet(CONFIG["model"]).to(DEVICE)
fdr_model.load_state_dict(_load_state(FDR_MODEL_PATH))
fdr_model.eval()
print("Models ready.\n")

# Combined score: FDR × exp(−ppm / PPM_LAMBDA)
# Smaller lambda = stronger ppm penalty
PPM_LAMBDA = 5.0

# ---------------------------------------------------------------------------
# Fetch caffeine spectra from GNPS via USI
# ---------------------------------------------------------------------------

USI_URL = "https://metabolomics-usi.gnps2.org/json/"
GNPS_META_URL = "https://gnps.ucsd.edu/ProteoSAFe/SpectrumCommentServlet"
DEFAULT_CE = "40"


def fetch_gnps_spectrum(accession: str) -> dict | None:
    """Fetch one spectrum from GNPS by accession ID. Returns spectrum dict or None."""
    usi = f"mzspec:GNPS:GNPS-LIBRARY:accession:{accession}"
    print(f"Fetching {accession} from GNPS …")
    try:
        resp = requests.get(USI_URL, params={"usi1": usi}, timeout=30)
        resp.raise_for_status()
        usi_data = resp.json()
    except Exception as e:
        print(f"  USI fetch failed: {e}")
        return None

    peaks = usi_data.get("peaks", [])
    precursor_mz = usi_data.get("precursor_mz")
    if not peaks or not precursor_mz:
        print("  No peaks/precursor_mz, skipping.")
        return None

    adduct, ce_str, instrument = "[M+H]+", DEFAULT_CE, "unknown"
    try:
        meta = requests.get(
            GNPS_META_URL, params={"SpectrumID": accession}, timeout=15
        ).json()
        ann = (meta.get("annotations") or [{}])[0]
        adduct = ann.get("Adduct") or adduct
        instrument = ann.get("Instrument") or instrument
        if ann.get("Collision_Energy") not in ("", "N/A", None, ""):
            ce_str = str(ann["Collision_Energy"])
        else:
            print(f"  WARNING: CE missing, using default {DEFAULT_CE} eV")
    except Exception as e:
        print(f"  Metadata fetch failed: {e}")

    print(
        f"  precursor_mz={precursor_mz}, adduct={adduct}, CE={ce_str}, "
        f"instrument={instrument}, n_peaks={len(peaks)}"
    )
    return {
        "title": accession,
        "source": "GNPS",
        "precursor_mz": float(precursor_mz),
        "precursor_type": adduct,
        "collision_energy": ce_str,
        "instrument": instrument,
        "peaks": peaks,
    }


# ---------------------------------------------------------------------------
# Load caffeine spectra from local filtered MGF files
# ---------------------------------------------------------------------------


def load_local_caffeine_spectra(mgf_path: str, titles: list[str]) -> list[dict]:
    """
    Read specific spectra (by TITLE) from a local filtered MGF file.
    Returns spectrum dicts in the same format as fetch_gnps_spectrum.
    """
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    wanted = set(titles)
    results = []
    current, peaks, in_block = {}, [], False

    with open(mgf_path) as f:
        for line in f:
            line = line.rstrip()
            if line == "BEGIN IONS":
                in_block, current, peaks = True, {}, []
            elif line == "END IONS":
                if in_block and current.get("title") in wanted:
                    # verify it's actually caffeine
                    smi = current.get("smiles", "")
                    try:
                        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
                        if mol and CalcMolFormula(mol) == "C8H10N4O2":
                            results.append(
                                {
                                    "title": current["title"],
                                    "source": os.path.basename(mgf_path)
                                    .replace("filtered_", "")
                                    .replace(".mgf", ""),
                                    "precursor_mz": float(
                                        current.get("precursor_mz", 0)
                                    ),
                                    "precursor_type": current.get(
                                        "precursor_type", "[M+H]+"
                                    ),
                                    "collision_energy": current.get(
                                        "collision_energy", DEFAULT_CE
                                    ),
                                    "instrument": current.get(
                                        "source_instrument", "unknown"
                                    ),
                                    "peaks": peaks,
                                }
                            )
                    except Exception:
                        pass
                in_block = False
            elif in_block:
                if "=" in line:
                    k, v = line.split("=", 1)
                    current[k.lower()] = v
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            peaks.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            pass
    return results


# ---------------------------------------------------------------------------
# Write a single-spectrum MGF
# ---------------------------------------------------------------------------


def write_mgf(spectrum: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("BEGIN IONS\n")
        f.write(f"TITLE={spectrum['title']}\n")
        f.write(f"PEPMASS={spectrum['precursor_mz']}\n")
        f.write(f"PRECURSOR_MZ={spectrum['precursor_mz']}\n")
        f.write(f"PRECURSOR_TYPE={spectrum['precursor_type']}\n")
        f.write(f"COLLISION_ENERGY={spectrum['collision_energy']}\n")
        for peak in spectrum["peaks"]:
            f.write(f"{peak[0]} {peak[1]}\n")
        f.write("END IONS\n")


# ---------------------------------------------------------------------------
# FDR reranking (mirrors run_fiddle.py)
# ---------------------------------------------------------------------------


def rerank_by_fdr(spec_t, env_t, refined_results, K):
    fdr_model.eval()
    refine_f = [f for f in refined_results["formula"] if f is not None]
    refine_m = [m for m in refined_results["mass"] if m is not None]
    if not refine_f:
        refined_results["fdr"] = [0.0] * K
        return refined_results

    f_vecs = [formula_to_vector(s) for s in refine_f]
    f_tensor = torch.from_numpy(np.array(f_vecs)).to(DEVICE, dtype=torch.float32)
    spec_rep = spec_t.to(DEVICE, dtype=torch.float32).repeat(f_tensor.size(0), 1)
    env_rep = env_t.to(DEVICE, dtype=torch.float32).repeat(f_tensor.size(0), 1)

    with torch.no_grad():
        fdr = fdr_model(spec_rep, env_rep, f_tensor)
        fdr = torch.sigmoid(fdr).detach().cpu().numpy()

    combined = sorted(zip(fdr, refine_f, refine_m), key=lambda x: x[0], reverse=True)
    sorted_fdr, sorted_f, sorted_m = map(list, zip(*combined))

    while len(sorted_f) < K:
        sorted_f.append(None)
        sorted_fdr.append(0.0)
        sorted_m.append(None)

    return {"formula": sorted_f, "mass": sorted_m, "fdr": sorted_fdr}


# ---------------------------------------------------------------------------
# Single-spectrum FIDDLE prediction
# ---------------------------------------------------------------------------


def predict(spectrum: dict) -> dict:
    """Run full FIDDLE pipeline on one spectrum dict. Returns results dict."""
    with tempfile.NamedTemporaryFile(suffix=".mgf", delete=False, mode="w") as tmp:
        mgf_path = tmp.name
    write_mgf(spectrum, mgf_path)

    try:
        dataset = MGFDataset(mgf_path, CONFIG["encoding"])
        if len(dataset) == 0:
            return {
                "error": "Spectrum filtered out (need ≥5 peaks, precursor 50–1500 Da)."
            }

        title, exp_pre_type, spec_arr, env_arr, neutral_add_arr = dataset[0]

        spec_t = (
            torch.from_numpy(np.array(spec_arr))
            .unsqueeze(0)
            .to(DEVICE, dtype=torch.float32)
        )
        env_t = (
            torch.from_numpy(np.array(env_arr))
            .unsqueeze(0)
            .to(DEVICE, dtype=torch.float32)
        )
        na_t = (
            torch.from_numpy(np.array(neutral_add_arr))
            .unsqueeze(0)
            .to(DEVICE, dtype=torch.float32)
        )

        t0 = time.time()
        with torch.no_grad():
            _, pred_f, pred_mass, _, _ = model(spec_t, env_t)
        pred_f -= na_t
        pred_time = time.time() - t0

        formula_init = vector_to_formula(pred_f[0])
        exp_pre_mz = env_t[0, 0].item()
        m = mass_calculator(exp_pre_type, exp_pre_mz)

        pp = CONFIG["post_processing"]
        refine_atom_type = list(pp["refine_atom_type"])
        refine_atom_num = list(pp["refine_atom_num"])

        refined = formula_refinement(
            [formula_init],
            m,
            pp["mass_tolerance"],
            pp["ppm_mode"],
            pp["top_k"],
            pp["maxium_miss_atom_num"],
            pp["time_out"],
            refine_atom_type,
            refine_atom_num,
        )

        t1 = time.time()
        refined = rerank_by_fdr(spec_t[0], env_t[0], refined, pp["top_k"])
        total_time = time.time() - t0

        predictions = []
        for i in range(pp["top_k"]):
            f = refined["formula"][i]
            mass_val = refined["mass"][i]
            if f is None or mass_val is None:
                continue
            ppm_error = abs(float(mass_val) - float(m)) / float(m) * 1e6
            fdr_score = float(refined["fdr"][i])
            combined = fdr_score * np.exp(-ppm_error / PPM_LAMBDA)
            predictions.append(
                {
                    "formula": f,
                    "mass": round(float(mass_val), 5),
                    "ppm_error": round(ppm_error, 2),
                    "confidence": round(fdr_score, 4),
                    "combined": round(combined, 4),
                }
            )

        return {
            "title": title,
            "precursor_mz": spectrum["precursor_mz"],
            "precursor_type": spectrum["precursor_type"],
            "experimental_mass": round(float(m), 5),
            "initial_prediction": formula_init,
            "predictions": predictions,
            "time_s": round(total_time, 3),
        }
    finally:
        os.unlink(mgf_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _marker(formula):
    return " <-- caffeine" if formula == "C8H10N4O2" else ""


def print_result(result: dict) -> None:
    if "error" in result:
        print(f"  Error: {result['error']}")
        return
    print(f"  Initial prediction : {result['initial_prediction']}")
    print(f"  Experimental mass  : {result['experimental_mass']} Da")
    print(f"  Processing time    : {result['time_s']} s")

    preds = result["predictions"]

    print(f"  FDR-ranked:")
    for i, p in enumerate(preds):
        print(
            f"    {i+1}. {p['formula']:<20}  ppm={p['ppm_error']:5.2f}  fdr={p['confidence']:.4f}{_marker(p['formula'])}"
        )

    combined = sorted(preds, key=lambda p: p["combined"], reverse=True)
    print(f"  Combined (FDR × exp(−ppm/{PPM_LAMBDA})):")
    for i, p in enumerate(combined):
        print(
            f"    {i+1}. {p['formula']:<20}  ppm={p['ppm_error']:5.2f}  combined={p['combined']:.4f}{_marker(p['formula'])}"
        )


if __name__ == "__main__":
    MGF_DIR = os.path.join(FIDDLE_DIR, "data", "mgf_debug")

    # Collect spectra from multiple sources
    all_spectra = []

    # 1. GNPS library spectrum (fetched via USI)
    gnps_spec = fetch_gnps_spectrum("CCMSLIB00016149314")
    if gnps_spec:
        all_spectra.append(gnps_spec)

    # 2. NIST20 orbitrap — pick one [M+H]+ spectrum per CE level
    nist20_titles = ["nist20_739832", "nist20_739834", "nist20_739836"]  # NCE=40/50/75%
    all_spectra += load_local_caffeine_spectra(
        os.path.join(MGF_DIR, "filtered_nist_orbitrap.mgf"), nist20_titles
    )

    # 3. NIST23 orbitrap — pick one [M+H]+ spectrum
    nist23_titles = ["nist23_731649", "nist23_731651", "nist23_731653"]  # NCE=40/50/75%
    all_spectra += load_local_caffeine_spectra(
        os.path.join(MGF_DIR, "filtered_nist23_orbitrap.mgf"), nist23_titles
    )

    # 4. MoNA orbitrap — the single caffeine spectrum (HCD NCE 40%)
    all_spectra += load_local_caffeine_spectra(
        os.path.join(MGF_DIR, "filtered_mona_orbitrap.mgf"), ["mona_orbitrap_38541"]
    )

    if not all_spectra:
        print("No usable caffeine spectra found.")
        sys.exit(1)

    for spec in all_spectra:
        print(f"\n{'='*60}")
        print(f"Spectrum  : {spec['title']}  [{spec['source']}]")
        print(f"Precursor : {spec['precursor_mz']} m/z  {spec['precursor_type']}")
        print(f"Instrument: {spec['instrument']}")
        print(f"CE        : {spec['collision_energy']}")
        print(f"Peaks     : {len(spec['peaks'])}")
        result = predict(spec)
        print_result(result)
