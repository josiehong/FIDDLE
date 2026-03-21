import re
import argparse
import yaml
from molmass import Formula
import time

from .mol_utils import ATOMS_WEIGHT, ATOMS_VALENCE


def parse_formula(formula):
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, formula)

    atom_counts = {}
    for atom, count in matches:
        count = int(count) if count else 1
        atom_counts[atom] = atom_counts.get(atom, 0) + count
    return atom_counts


def format_formula(atom_counts):
    # Sort atoms alphabetically and format the formula
    return "".join(
        [
            atom + (str(count) if count > 1 else "")
            for atom, count in sorted(atom_counts.items())
        ]
    )


def passes_senior_rule(formula):
    """Check whether a molecular formula satisfies the SENIOR valence rules.

    The three SENIOR rules are:
      1. The sum of odd-valence atom counts must be even.
      2. Total valence >= 2 * max single-atom valence.
      3. Total valence >= 2 * (number of atoms - 1).

    Args:
            formula: Molecular formula string, e.g. 'C6H12O6'.

    Returns:
            bool: True if all three rules are satisfied.
    """
    atom_counts = parse_formula(formula)
    total_valence = 0
    max_valence = 0

    for atom, count in atom_counts.items():
        valence = ATOMS_VALENCE.get(atom, 0)
        total_valence += valence * count
        max_valence = max(max_valence, valence)

    num_atoms = sum(atom_counts.values())

    sum_odd_valences = sum(
        valence * count
        for atom, count in atom_counts.items()
        if ATOMS_VALENCE.get(atom, 0) % 2 != 0
    )
    if sum_odd_valences % 2 != 0:
        return False

    if total_valence < 2 * max_valence:
        return False

    if total_valence < 2 * (num_atoms - 1):
        return False

    return True


def exceed_refine_atom_limit(
    refined_counts, formulas, refine_atom_type, refine_atom_num
):
    """Check if the refined formula exceeds the refinement limit for a given atom type."""
    # refined_counts = parse_formula(refined_formula)
    for formula in formulas:
        original_counts = parse_formula(formula)

        for t, n in zip(refine_atom_type, refine_atom_num):
            if n == -1:  # No limitation
                continue

            refined_n = refined_counts.get(t, 0)
            original_n = original_counts.get(t, 0)
            # Check if the difference in atom counts exceeds the allowed number
            if abs(refined_n - original_n) > n:
                return True  # Exceeds the limit for at least one formula

    return False  # Does not exceed the limit for any formula


def adjust_hydrogen(atom_counts, M):
    delta_h = round((M - Formula(format_formula(atom_counts)).isotope.mass) / 1.007941)
    if "H" not in atom_counts.keys():
        atom_counts["H"] = delta_h
    else:
        atom_counts["H"] += delta_h
    if atom_counts["H"] <= 0:
        del atom_counts["H"]
    return atom_counts


def remove_duplicates_preserve_order(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def candidate_formulas_generation(f, M, f0_list, refine_atom_type, refine_atom_num):
    """Generate neighboring formula candidates by adding or removing one heavy atom.

    For each non-H atom in refine_atom_type, produces two candidates: one with the
    atom count incremented by 1 and one decremented by 1. After each change,
    hydrogen count is adjusted to match the target mass M. Candidates that exceed
    the per-atom refinement limits relative to any formula in f0_list are excluded.
    Results are sorted by proximity to M.

    Args:
            f: Current formula string to expand neighbors from.
            M: Target monoisotopic mass in Da.
            f0_list: List of initial predicted formula strings (used for limit checking).
            refine_atom_type: List of atom symbols to vary, e.g. ['C', 'N', 'O', ...].
            refine_atom_num: Per-atom maximum deviation allowed; -1 means no limit.

    Returns:
            list[str]: Candidate formula strings sorted by |mass - M|.
    """
    atom_counts = parse_formula(f)
    formulas = []

    for a in refine_atom_type:
        if a == "H":
            continue  # Skip 'H'

        # Add one atom
        new_atom_counts = dict(atom_counts)
        new_atom_counts[a] = new_atom_counts.get(a, 0) + 1
        if (
            not exceed_refine_atom_limit(
                new_atom_counts, f0_list, refine_atom_type, refine_atom_num
            )
            and "H" in atom_counts.keys()
        ):
            new_atom_counts = adjust_hydrogen(new_atom_counts, M)  # Ajust 'H'
            formulas.append(format_formula(new_atom_counts))

        # Remove one atom if it exists
        if a in atom_counts and atom_counts[a] > 0:
            new_atom_counts = dict(atom_counts)
            new_atom_counts[a] -= 1
            if new_atom_counts[a] == 0:
                del new_atom_counts[a]
            if (
                not exceed_refine_atom_limit(
                    new_atom_counts, f0_list, refine_atom_type, refine_atom_num
                )
                and "H" in atom_counts.keys()
            ):
                new_atom_counts = adjust_hydrogen(new_atom_counts, M)  # Ajust 'H'
                formulas.append(format_formula(new_atom_counts))

    # Sort based on proximity to M
    formulas.sort(key=lambda formula: abs(Formula(formula).isotope.mass - M))

    return formulas


def formula_refinement(
    f0_list, M, delta_M, ppm_mode, K, D, T, refine_atom_type, refine_atom_num
):
    """Refine predicted formulas via a best-first search around the target mass.

    Starting from one or more predicted formulas (f0_list), performs a guided
    search by iteratively expanding neighbors (±1 heavy atom, H adjusted to fit M).
    Candidates are accepted when their monoisotopic mass falls within delta_M of M
    and they pass the SENIOR valence rules. Search depth is bounded by D, and the
    search exits early if K results are found or the timeout T is exceeded.

    Args:
            f0_list:         List of initial predicted formula strings.
            M:               Target monoisotopic mass in Da.
            delta_M:         Mass tolerance (Da if ppm_mode=False, ppm if ppm_mode=True).
            ppm_mode:        If True, interpret delta_M as ppm and convert to Da using M.
            K:               Maximum number of refined formulas to return.
            D:               Maximum search depth (number of atom-change steps from f0).
            T:               Timeout in seconds; set to 0 to disable.
            refine_atom_type: List of atom symbols to vary during search.
            refine_atom_num:  Per-atom maximum deviation from f0; -1 means no limit.

    Returns:
            dict: {'formula': list[str | None], 'mass': list[float | None]}, each of
                  length K, padded with None if fewer than K formulas are found.
                  Results are sorted by |mass - M|.
    """
    start_time = time.time()
    refine_f = []  # List of refined formulas
    trace_f = set()  # Keep track of all formulas that have been focused on

    if ppm_mode:
        delta_M = delta_M * M / 10**6

    # If no halogen in prediction, no halogen in refinement
    # pred_atom_types = set()
    # for f0 in f0_list:
    # 	pred_atom_types.update([atom for atom, n in parse_formula(f0).items() if n > 0])
    # for atom in ['P', 'S', 'F', 'Cl', 'B', 'Br', 'I', 'Na', 'K']:
    # 	if atom not in pred_atom_types and atom in refine_atom_type:
    # 		i = refine_atom_type.index(atom)
    # 		refine_atom_type.pop(i)
    # 		refine_atom_num.pop(i)

    candidate_f = [
        format_formula(adjust_hydrogen(parse_formula(f0), M)) for f0 in f0_list
    ]  # Initialize candidate_f
    candidate_d = [0] * len(candidate_f)
    search_deep = 0
    # print('Init {} candidate formulas from predictions'.format(len(candidate_f)))
    # print(candidate_f)
    # print([(Formula(f).isotope.mass, abs(Formula(f).isotope.mass - M)) for f in candidate_f])

    # Search >>>
    while len(refine_f) < K and (
        candidate_f or search_deep == 0
    ):  # It is okay if at begining there is no formula in candidate_f
        if search_deep == 0:
            # Check if the predicted formula falls into the target space
            for f0 in candidate_f:
                m = Formula(f0).isotope.mass
                if (
                    M - delta_M <= m <= M + delta_M
                    and passes_senior_rule(f0)
                    and f0 not in refine_f
                ):  # Passed SENIOR rule
                    refine_f.append(f0)  # Append the refined formula for return
                    # print('Got one refined formula')

            # Generate more candidates from the init formula list
            new_candidates = []
            for f0 in f0_list:
                new_candidates += candidate_formulas_generation(
                    f0, M, f0_list, refine_atom_type, refine_atom_num
                )
            new_candidates = remove_duplicates_preserve_order(
                new_candidates
            )  # Remove duplicates
            new_candidates.sort(
                key=lambda formula: abs(Formula(formula).isotope.mass - M)
            )  # Need resort, because they are extend from multiple candidate list
            new_depths = [search_deep + 1] * len(new_candidates)

            candidate_f = new_candidates + candidate_f
            candidate_d = new_depths + candidate_d

        elif search_deep <= D:
            # Generate more candidates from the current focus formula f
            new_candidates = candidate_formulas_generation(
                f, M, f0_list, refine_atom_type, refine_atom_num
            )
            new_candidates = remove_duplicates_preserve_order(
                new_candidates
            )  # Remove duplicates
            new_candidates = [
                i for i in new_candidates if i not in trace_f and i not in candidate_f
            ]
            new_depths = [search_deep + 1] * len(new_candidates)

            while len(new_candidates) == 0 and len(candidate_f) > 0:
                f = candidate_f.pop(
                    0
                )  # Reject to continue (no new candidate), trace back
                search_deep = candidate_d.pop(0)
                trace_f.add(f)
                # print('Trackback')

                # Generate more candidates from the current focus formula f
                new_candidates = candidate_formulas_generation(
                    f, M, f0_list, refine_atom_type, refine_atom_num
                )
                new_candidates = remove_duplicates_preserve_order(
                    new_candidates
                )  # Remove duplicates
                new_candidates = [
                    i
                    for i in new_candidates
                    if i not in trace_f and i not in candidate_f
                ]
                new_depths = [search_deep + 1] * len(new_candidates)
            if len(new_candidates) == 0:  # No other candidates exit
                break

            candidate_f = new_candidates + candidate_f
            candidate_d = new_depths + candidate_d

        else:  # While exceed search limitation, the candidate_f will be depleted
            if len(candidate_f) == 0:
                break
            f = candidate_f.pop(
                0
            )  # Reject to continue (exceed search limitation), trace back
            search_deep = candidate_d.pop(0)
            trace_f.add(f)
            # print('Trackback')

        if len(candidate_f) == 0:
            break
        f = candidate_f.pop(0)  # Select the next formula to focus on
        search_deep = candidate_d.pop(0)
        m = Formula(f).isotope.mass
        trace_f.add(f)
        # print('\nNext step: {} ({}, {})'.format(f, m, abs(m - M)))

        if M - delta_M <= m <= M + delta_M:
            if passes_senior_rule(f) and f not in refine_f:  # Passed SENIOR rule
                refine_f.append(f)  # Append the refined formula for return
                # print('Got one refined formula')

        # Check if timeout
        running_time = time.time() - start_time
        if T > 0 and running_time > T:  # Timeout
            break
    # End >>>

    refine_f.sort(key=lambda formula: abs(Formula(formula).isotope.mass - M))
    refine_m = [Formula(formula).isotope.mass for formula in refine_f]

    # Pad refine_f and refine_m with None if necessary
    if len(refine_f) < K:
        refine_f += [None] * (K - len(refine_f))
        refine_m += [None] * (K - len(refine_m))

    return {"formula": refine_f, "mass": refine_m}


if __name__ == "__main__":
    # Post-process settings
    # to check `formula_refinement`: python refine_utils.py --config_path ./config/fiddle_tcn.yml
    parser = argparse.ArgumentParser(
        description="Mass Spectra to formula (post-process)"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to configuration (.yml)"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    refine_atom_type = config["post_processing"]["refine_atom_type"]
    refine_atom_num = config["post_processing"]["refine_atom_num"]
    mass_tolerance = config["post_processing"]["mass_tolerance"]
    ppm_mode = config["post_processing"]["ppm_mode"]
    top_k = config["post_processing"]["top_k"]
    maxium_miss_atom_num = config["post_processing"]["maxium_miss_atom_num"]
    time_out = config["post_processing"]["time_out"]

    predicted_formula = ["C6H7O2N2"]
    target_formula = "C5H9NO4"
    compound_mass = Formula(target_formula).isotope.mass

    print("predicted formulas:", predicted_formula)
    print("target formula: {} ({})".format(target_formula, compound_mass))

    result = formula_refinement(
        predicted_formula,
        compound_mass,
        mass_tolerance,
        ppm_mode,
        top_k,
        maxium_miss_atom_num,
        time_out,
        refine_atom_type,
        refine_atom_num,
    )
    print("restults:", result)
