#!/usr/bin/env python3
"""
Test FIDDLE formula prediction on caffeine (C8H10N4O2) using the Q-TOF model.

Spectra tested
--------------
- GNPS CCMSLIB00016149314 (fetched live via USI)
- NIST20 Q-TOF spectra (CE = 10 / 20 / 27 %)
- MoNA Q-TOF spectra (various CE)
- Agilent combined Q-TOF spectra

Caffeine InChIKey: RYYVLZVUVIJVGH-UHFFFAOYSA-N
"""

import os
import sys
import tempfile
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

FIDDLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIDDLE_DIR)

from dataset import MGFDataset
from model_tcn import MS2FNet_tcn, FormulaEncoder, RescoreHead
from utils import (
    formula_refinement,
    formula_to_dict,
    formula_to_vector,
    mass_calculator,
    vector_to_formula,
)

DEVICE = torch.device("cpu")

CONFIG_PATH = os.path.join(FIDDLE_DIR, "config", "fiddle_tcn_qtof.yml")
TCN_PATH = os.path.join(FIDDLE_DIR, "check_point", "fiddle_tcn_qtof_031826.pt")
RESCORE_PATH = os.path.join(FIDDLE_DIR, "check_point", "fiddle_rescore_qtof_031826.pt")

MGF_DIR = os.path.join(FIDDLE_DIR, "data", "mgf_debug")

DEFAULT_CE = "40"

# Local caffeine spectrum titles per source
NIST_QTOF_TITLES = ["nist20_20310", "nist20_20313", "nist20_20316"]
MONA_QTOF_TITLES = [
    "mona_qtof_25485",
    "mona_qtof_25486",
    "mona_qtof_25487",
    "mona_qtof_33932",
]
AGILENT_TITLES = [
    "agilent_combine_16753",
    "agilent_combine_16754",
    "agilent_combine_16755",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_state(path):
    sd = torch.load(path, map_location=DEVICE, weights_only=False)["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = OrderedDict((k.removeprefix("module."), v) for k, v in sd.items())
    return sd


def load_models():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Loading Q-TOF models …")
    tcn_model = MS2FNet_tcn(config["model"]).to(DEVICE)
    tcn_model.load_state_dict(_load_state(TCN_PATH))
    tcn_model.eval()

    rescore_formula_encoder = None
    rescore_head = None
    if os.path.exists(RESCORE_PATH):
        ckpt = torch.load(RESCORE_PATH, map_location=DEVICE)
        rescore_formula_encoder = FormulaEncoder(config["model"]).to(DEVICE)
        rescore_formula_encoder.load_state_dict(ckpt["formula_encoder_state_dict"])
        rescore_formula_encoder.eval()
        rescore_head = RescoreHead(config["model"]).to(DEVICE)
        rescore_head.load_state_dict(ckpt["rescore_head_state_dict"])
        rescore_head.eval()
        print(f"  Loaded rescore model from {RESCORE_PATH}")
    else:
        print(
            f"  WARNING: Rescore model not found at {RESCORE_PATH}. Rescore scores will be 0."
        )

    print("  Q-TOF models ready.")
    return config, tcn_model, rescore_formula_encoder, rescore_head


# ---------------------------------------------------------------------------
# Spectrum loading
# ---------------------------------------------------------------------------


def load_local_spectra(mgf_path, titles, source_name):
    """Load spectra with the given titles from a local MGF file."""
    if not os.path.exists(mgf_path):
        print(f"  WARNING: {mgf_path} not found, skipping.")
        return []

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
                    results.append(
                        {
                            "title": current["title"],
                            "source": source_name,
                            "precursor_mz": float(current.get("precursor_mz", 0)),
                            "precursor_type": current.get("precursor_type", "[M+H]+"),
                            "collision_energy": current.get(
                                "collision_energy", DEFAULT_CE
                            ),
                            "instrument": current.get("source_instrument", "unknown"),
                            "peaks": peaks,
                        }
                    )
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
# Prediction
# ---------------------------------------------------------------------------


def write_mgf(spectrum, path):
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


def rescore_candidates(
    z_spec, refined_results, K, rescore_formula_encoder, rescore_head
):
    refine_f = [f for f in refined_results["formula"] if f is not None]
    refine_m = [m for m in refined_results["mass"] if m is not None]
    if not refine_f or rescore_formula_encoder is None:
        refined_results["rescore"] = [0.0] * K
        return refined_results

    f_vecs = torch.tensor(
        [formula_to_vector(s) for s in refine_f], dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        z_form = rescore_formula_encoder(f_vecs)
        z_spec_rep = z_spec.unsqueeze(0).expand(z_form.size(0), -1)
        logits = rescore_head(z_spec_rep * z_form)
        rescore_scores = torch.sigmoid(logits).cpu().numpy()

    ranked = sorted(
        zip(rescore_scores, refine_f, refine_m),
        key=lambda x: x[0],
        reverse=True,
    )
    sorted_rescore, sorted_f, sorted_m = map(list, zip(*ranked))

    while len(sorted_f) < K:
        sorted_f.append(None)
        sorted_rescore.append(0.0)
        sorted_m.append(None)

    return {"formula": sorted_f, "mass": sorted_m, "rescore": sorted_rescore}


def predict(spectrum, config, tcn_model, rescore_formula_encoder, rescore_head):
    with tempfile.NamedTemporaryFile(suffix=".mgf", delete=False, mode="w") as tmp:
        mgf_path = tmp.name
    write_mgf(spectrum, mgf_path)

    try:
        dataset = MGFDataset(mgf_path, config["encoding"])
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
            _, pred_f, pred_mass, _, _ = tcn_model(spec_t, env_t)
        pred_f -= na_t

        env_rescore = env_t.clone()
        env_rescore[:, 0] = 0.0
        with torch.no_grad():
            encoded_x_rescore, _, _, _, _ = tcn_model(spec_t, env_rescore)
        z_spec = F.normalize(encoded_x_rescore[0], dim=0)

        formula_init = vector_to_formula(pred_f[0])
        exp_pre_mz = env_t[0, 0].item()
        m = mass_calculator(exp_pre_type, exp_pre_mz)

        pp = config["post_processing"]
        refine_atom_type = list(pp["refine_atom_type"])
        refine_atom_num = list(pp["refine_atom_num"])

        for atom, cnt in formula_to_dict(formula_init).items():
            if atom == "H" or atom in refine_atom_type:
                continue
            refine_atom_type.append(atom)
            refine_atom_num.append(max(1, int(cnt)))

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

        refined = rescore_candidates(
            z_spec, refined, pp["top_k"], rescore_formula_encoder, rescore_head
        )
        total_time = time.time() - t0

        predictions = []
        for i in range(pp["top_k"]):
            f = refined["formula"][i]
            mass_val = refined["mass"][i]
            if f is None or mass_val is None:
                continue
            ppm_error = abs(float(mass_val) - float(m)) / float(m) * 1e6
            rescore_score = float(refined["rescore"][i])
            predictions.append(
                {
                    "formula": f,
                    "mass": round(float(mass_val), 5),
                    "ppm_error": round(ppm_error, 2),
                    "rescore": round(rescore_score, 4),
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
# Display
# ---------------------------------------------------------------------------


def _marker(formula):
    return " <-- caffeine" if formula == "C8H10N4O2" else ""


def print_result(result):
    if "error" in result:
        print(f"  Error: {result['error']}")
        return
    print(f"  Initial prediction : {result['initial_prediction']}")
    print(f"  Experimental mass  : {result['experimental_mass']} Da")
    print(f"  Processing time    : {result['time_s']} s")

    preds = result["predictions"]
    print("  Rescore-ranked:")
    for i, p in enumerate(preds):
        print(
            f"    {i+1}. {p['formula']:<20}  ppm={p['ppm_error']:5.2f}"
            f"  rescore={p['rescore']:.4f}{_marker(p['formula'])}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config, tcn_model, rescore_fe, rescore_hd = load_models()

    all_spectra = []

    all_spectra += load_local_spectra(
        os.path.join(MGF_DIR, "filtered_nist_qtof.mgf"), NIST_QTOF_TITLES, "nist_qtof"
    )
    all_spectra += load_local_spectra(
        os.path.join(MGF_DIR, "filtered_mona_qtof.mgf"), MONA_QTOF_TITLES, "mona_qtof"
    )
    all_spectra += load_local_spectra(
        os.path.join(MGF_DIR, "filtered_agilent_qtof.mgf"),
        AGILENT_TITLES,
        "agilent_qtof",
    )

    if not all_spectra:
        print("No usable caffeine spectra found.")
        sys.exit(1)

    print(f"\nLoaded {len(all_spectra)} caffeine spectra from Q-TOF sources.\n")

    for spec in all_spectra:
        print(f"\n{'='*60}")
        print(f"Spectrum  : {spec['title']}  [{spec['source']}]")
        print(f"Precursor : {spec['precursor_mz']} m/z  {spec['precursor_type']}")
        print(f"Instrument: {spec['instrument']}")
        print(f"CE        : {spec['collision_energy']}")
        print(f"Peaks     : {len(spec['peaks'])}")
        result = predict(spec, config, tcn_model, rescore_fe, rescore_hd)
        print_result(result)
