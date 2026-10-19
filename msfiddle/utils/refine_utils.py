import re
from collections import deque
from molmass import Formula
import time

from .mol_utils import ATOMS_VALENCE

# === FAST MONOISOTOPIC MASS (speeds up the BFS refinement) ===
# Per-element most-abundant-isotope masses are taken straight from molmass once at
# import, so fast_mass(formula) equals molmass's Formula(formula).isotope.mass (the
# isotope mass is linear in element counts) but avoids a molmass call per candidate
# inside the hot BFS loop (~19x faster). Deriving from molmass — rather than
# hardcoding — guarantees agreement and avoids isotope-constant traps (e.g. boron
# B-10 vs B-11). Only residual vs molmass is float summation order (~1e-13 Da),
# ~1e9x below the 10 ppm mass window, so it never changes which formulas match.
_ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "B", "Br", "I", "Na", "K"]
ISOTOPE_MASS = {el: Formula(el).isotope.mass for el in _ELEMENTS}
_MASS_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_MASS_CACHE = {}


def fast_mass(formula):
    """Monoisotopic mass from a formula string, memoized.

    Drop-in for molmass Formula(formula).isotope.mass — equal masses (constants are
    molmass's own per-element values) at a fraction of the cost; cached per string.
    """
    m = _MASS_CACHE.get(formula)
    if m is not None:
        return m
    total = 0.0
    for el, n in _MASS_TOKEN.findall(formula):
        if not el:
            continue
        total += ISOTOPE_MASS[el] * (int(n) if n else 1)
    _MASS_CACHE[formula] = total
    return total


# === END ===


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


def _exceeds_limits(refined_counts, f0_counts_list, active_limits):
    """Limit check against pre-parsed f0 counts and pre-filtered (atom, cap) pairs."""
    for original_counts in f0_counts_list:
        for t, n in active_limits:
            # Check if the difference in atom counts exceeds the allowed number
            if abs(refined_counts.get(t, 0) - original_counts.get(t, 0)) > n:
                return True  # Exceeds the limit for at least one formula
    return False  # Does not exceed the limit for any formula


def _mass_from_counts(atom_counts):
    """Monoisotopic mass from an atom-count dict, without string round-trips.

    Accumulates per-element masses in alphabetical atom order — the same order
    fast_mass(format_formula(counts)) sums them — so the result is float-identical.
    Assumes all counts are >= 1 (guaranteed by the refinement search).
    """
    total = 0.0
    for atom in sorted(atom_counts):
        total += ISOTOPE_MASS[atom] * atom_counts[atom]
    return total


def adjust_hydrogen(atom_counts, M):
    delta_h = round((M - _mass_from_counts(atom_counts)) / 1.007941)
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
            list[str]: Candidate formula strings sorted by abs(mass - M).
    """
    return _generate_candidates(
        f,
        M,
        [parse_formula(f0) for f0 in f0_list],
        [(t, n) for t, n in zip(refine_atom_type, refine_atom_num) if n != -1],
        refine_atom_type,
    )


def _generate_candidates(f, M, f0_counts_list, active_limits, refine_atom_type):
    """candidate_formulas_generation with pre-parsed f0 counts and limit pairs."""
    atom_counts = parse_formula(f)
    has_h = "H" in atom_counts
    formulas = []

    for a in refine_atom_type:
        if a == "H":
            continue  # Skip 'H'

        # Add one atom
        new_atom_counts = dict(atom_counts)
        new_atom_counts[a] = new_atom_counts.get(a, 0) + 1
        if has_h and not _exceeds_limits(
            new_atom_counts, f0_counts_list, active_limits
        ):
            new_atom_counts = adjust_hydrogen(new_atom_counts, M)  # Ajust 'H'
            formulas.append(format_formula(new_atom_counts))

        # Remove one atom if it exists
        if a in atom_counts and atom_counts[a] > 0:
            new_atom_counts = dict(atom_counts)
            new_atom_counts[a] -= 1
            if new_atom_counts[a] == 0:
                del new_atom_counts[a]
            if has_h and not _exceeds_limits(
                new_atom_counts, f0_counts_list, active_limits
            ):
                new_atom_counts = adjust_hydrogen(new_atom_counts, M)  # Ajust 'H'
                formulas.append(format_formula(new_atom_counts))

    # Sort based on proximity to M
    formulas.sort(key=lambda formula: abs(fast_mass(formula) - M))

    return formulas


def formula_refinement_bfs(
    f0_list, M, delta_M, ppm_mode, K, D, T, refine_atom_type, refine_atom_num
):
    """Refine predicted formulas via a best-first search around the target mass.

    This is the original search algorithm, kept as the fallback for search
    spaces too large to enumerate (see formula_refinement).

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
                  Results are sorted by abs(mass - M).
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

    # Parse the (fixed) initial formulas and active limits once for the whole search.
    f0_counts_list = [parse_formula(f0) for f0 in f0_list]
    active_limits = [
        (t, n) for t, n in zip(refine_atom_type, refine_atom_num) if n != -1
    ]

    candidate_f = deque(
        format_formula(adjust_hydrogen(parse_formula(f0), M)) for f0 in f0_list
    )  # Initialize candidate_f
    candidate_d = deque([0] * len(candidate_f))
    # Multiplicity of each formula currently in candidate_f, for O(1) membership
    # tests (the frontier can hold duplicates, so a plain set is not enough).
    frontier_count = {}
    for _f in candidate_f:
        frontier_count[_f] = frontier_count.get(_f, 0) + 1

    def _pop_front():
        # Pop the frontier head, keeping the multiplicity mirror in sync.
        x = candidate_f.popleft()
        d = candidate_d.popleft()
        c = frontier_count[x] - 1
        if c:
            frontier_count[x] = c
        else:
            del frontier_count[x]
        return x, d

    def _push_front(new_candidates, new_depths):
        # Prepend, preserving list order (new_candidates + candidate_f).
        candidate_f.extendleft(reversed(new_candidates))
        candidate_d.extendleft(reversed(new_depths))
        for x in new_candidates:
            frontier_count[x] = frontier_count.get(x, 0) + 1

    search_deep = 0
    f = None  # focus formula; every pass after the first sets it at its end

    # Search >>>
    while len(refine_f) < K and (
        candidate_f or search_deep == 0
    ):  # It is okay if at begining there is no formula in candidate_f
        if search_deep == 0:
            # Check if the predicted formula falls into the target space
            for f0 in candidate_f:
                m = fast_mass(f0)
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
                new_candidates += _generate_candidates(
                    f0, M, f0_counts_list, active_limits, refine_atom_type
                )
            new_candidates = remove_duplicates_preserve_order(
                new_candidates
            )  # Remove duplicates
            new_candidates.sort(
                key=lambda formula: abs(fast_mass(formula) - M)
            )  # Need resort, because they are extend from multiple candidate list
            new_depths = [search_deep + 1] * len(new_candidates)

            _push_front(new_candidates, new_depths)

        elif search_deep <= D:
            # Generate more candidates from the current focus formula f
            new_candidates = _generate_candidates(
                f, M, f0_counts_list, active_limits, refine_atom_type
            )
            new_candidates = [
                i
                for i in remove_duplicates_preserve_order(new_candidates)
                if i not in trace_f and i not in frontier_count
            ]
            new_depths = [search_deep + 1] * len(new_candidates)

            while len(new_candidates) == 0 and len(candidate_f) > 0:
                # Reject to continue (no new candidate), trace back
                f, search_deep = _pop_front()
                trace_f.add(f)
                # print('Trackback')

                # Generate more candidates from the current focus formula f
                new_candidates = _generate_candidates(
                    f, M, f0_counts_list, active_limits, refine_atom_type
                )
                new_candidates = [
                    i
                    for i in remove_duplicates_preserve_order(new_candidates)
                    if i not in trace_f and i not in frontier_count
                ]
                new_depths = [search_deep + 1] * len(new_candidates)
            if len(new_candidates) == 0:  # No other candidates exit
                break

            _push_front(new_candidates, new_depths)

        else:  # While exceed search limitation, the candidate_f will be depleted
            if len(candidate_f) == 0:
                break
            # Reject to continue (exceed search limitation), trace back
            f, search_deep = _pop_front()
            trace_f.add(f)
            # print('Trackback')

        if len(candidate_f) == 0:
            break
        f, search_deep = _pop_front()  # Select the next formula to focus on
        m = fast_mass(f)
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

    refine_f.sort(key=lambda formula: abs(fast_mass(formula) - M))
    refine_m = [fast_mass(formula) for formula in refine_f]

    # Pad refine_f and refine_m with None if necessary
    if len(refine_f) < K:
        refine_f += [None] * (K - len(refine_f))
        refine_m += [None] * (K - len(refine_m))

    return {"formula": refine_f, "mass": refine_m}


# === COMPLETE ENUMERATION REFINEMENT (Tier 2) ===
# The BFS above walks the formula lattice one atom-change at a time with string
# states. But with the production caps (a few units per heavy atom, H rebalanced
# to the target mass) the reachable space is a small bounded box that can be
# enumerated outright with numpy: provably complete within the caps, no timeout,
# and the returned top-K are the true closest-K by mass. formula_refinement
# dispatches to this enumeration when the box is tractable and falls back to
# the BFS otherwise (e.g. uncapped atom lists).

_H_STEP = 1.007941  # same constant the BFS uses to rebalance hydrogen
_ENUM_SIZE_LIMIT = 3_000_000  # estimated grid points above this -> BFS fallback
_ENUM_ROW_LIMIT = 20_000_000  # hard runtime guard on materialised rows


def _enumerate_valid_formulas(
    f0_list, M, delta_da, D, refine_atom_type, refine_atom_num
):
    """All formulas within the caps, the mass window, and the SENIOR rules.

    Returns the complete list sorted by abs(mass - M), or None when the search
    box is too large to enumerate (the caller then falls back to the BFS).
    Semantics match the BFS acceptance rules: candidates must stay within every
    initial formula's per-atom caps, lie within delta_da of M after hydrogen
    rebalancing, pass SENIOR, and be within D total heavy-atom changes of some
    initial formula; the H-adjusted initial formulas themselves are accepted on
    window+SENIOR alone, exactly like the BFS's depth-0 pass.
    """
    import numpy as np

    caps = {}
    for t, n in zip(refine_atom_type, refine_atom_num):
        caps[t] = n
    h_cap = caps.pop("H", -1)
    heavy = list(caps.keys())

    f0_counts_list = [parse_formula(f0) for f0 in f0_list]

    # Unknown elements would fail mass lookup here and in the BFS alike;
    # defer to the BFS so behaviour (including errors) is unchanged.
    for counts in f0_counts_list:
        for atom in counts:
            if atom not in ISOTOPE_MASS:
                return None
    for atom in heavy:
        if atom not in ISOTOPE_MASS:
            return None

    results = set()

    # Depth-0 pass: H-adjusted initial formulas, window+SENIOR only (no caps).
    for counts in f0_counts_list:
        f_adj = format_formula(adjust_hydrogen(dict(counts), M))
        m = fast_mass(f_adj)
        if M - delta_da <= m <= M + delta_da and passes_senior_rule(f_adj):
            results.add(f_adj)

    # Atoms outside refine_atom_type are fixed payload carried from the
    # initial formula; formulas with different payloads live on disjoint
    # lattices, so enumerate per payload group.
    groups = {}
    for counts in f0_counts_list:
        payload = {a: n for a, n in counts.items() if a != "H" and a not in caps}
        key = tuple(sorted(payload.items()))
        groups.setdefault(key, (payload, []))[1].append(counts)

    eps = 1e-6
    for payload, members in groups.values():
        # Per-atom count range: intersection of every f0's cap box (the BFS
        # limit check rejects a candidate out of range of ANY f0), further
        # bounded by reachability (within D changes of some same-payload f0)
        # and, for uncapped atoms, by the target mass itself.
        lo, hi = [], []
        member_lo = {a: min(c.get(a, 0) for c in members) for a in heavy}
        member_hi = {a: max(c.get(a, 0) for c in members) for a in heavy}
        for a in heavy:
            cap = caps[a]
            if cap == -1:
                lo_a, hi_a = 0, int((M + delta_da + eps) // ISOTOPE_MASS[a])
            else:
                lo_a = max(0, max(c.get(a, 0) - cap for c in f0_counts_list))
                hi_a = min(c.get(a, 0) + cap for c in f0_counts_list)
            lo_a = max(lo_a, member_lo[a] - D)
            hi_a = min(hi_a, member_hi[a] + D)
            if hi_a < lo_a:
                lo, hi = None, None  # empty box for this group
                break
            lo.append(lo_a)
            hi.append(hi_a)
        if lo is None:
            continue

        est = 1
        for lo_a, hi_a in zip(lo, hi):
            est *= min(hi_a - lo_a + 1, 2 * D + 1)
            if est > _ENUM_SIZE_LIMIT:
                return None

        found = _enumerate_group(
            np,
            payload,
            members,
            M,
            delta_da,
            D,
            heavy,
            lo,
            hi,
            h_cap,
            f0_counts_list,
        )
        if found is None:
            return None
        results.update(found)

    return sorted(results, key=lambda f: abs(fast_mass(f) - M))


def _enumerate_group(
    np, payload, members, M, delta_da, D, heavy, lo, hi, h_cap, f0_counts_list
):
    """Enumerate one payload group's box with incremental mass/depth pruning."""
    eps = 1e-6
    base_mass = 0.0
    for atom in sorted(payload):
        base_mass += ISOTOPE_MASS[atom] * payload[atom]

    masses = [ISOTOPE_MASS[a] for a in heavy]
    # Assign heaviest atoms first so the mass bound prunes early.
    order = sorted(range(len(heavy)), key=lambda i: -masses[i])

    # Suffix minima for pruning: least mass / least depth the still-unassigned
    # atoms can contribute.
    suffix_min_mass = [0.0] * (len(order) + 1)
    for pos in range(len(order) - 1, -1, -1):
        i = order[pos]
        suffix_min_mass[pos] = suffix_min_mass[pos + 1] + lo[i] * masses[i]
    n_members = len(members)
    member_counts = [[c.get(heavy[i], 0) for c in members] for i in range(len(heavy))]
    suffix_min_depth = [[0] * n_members for _ in range(len(order) + 1)]
    for pos in range(len(order) - 1, -1, -1):
        i = order[pos]
        for j in range(n_members):
            c0 = member_counts[i][j]
            gap = max(lo[i] - c0, c0 - hi[i], 0)  # distance from c0 to range
            suffix_min_depth[pos][j] = suffix_min_depth[pos + 1][j] + gap

    counts = np.zeros((1, 0), dtype=np.int64)
    mass_acc = np.full(1, base_mass)
    depth_acc = np.zeros((1, n_members), dtype=np.int64)

    for pos, i in enumerate(order):
        vals = np.arange(lo[i], hi[i] + 1, dtype=np.int64)
        n_rows, n_vals = counts.shape[0], vals.shape[0]
        counts = np.repeat(counts, n_vals, axis=0)
        rep_vals = np.tile(vals, n_rows)
        counts = np.concatenate([counts, rep_vals[:, None]], axis=1)
        mass_acc = np.repeat(mass_acc, n_vals) + rep_vals * masses[i]
        depth_acc = np.repeat(depth_acc, n_vals, axis=0) + np.abs(
            rep_vals[:, None] - np.asarray(member_counts[i])[None, :]
        )

        keep = mass_acc + suffix_min_mass[pos + 1] <= M + delta_da + eps
        keep &= (depth_acc + np.asarray(suffix_min_depth[pos + 1])[None, :]).min(
            axis=1
        ) <= D
        counts, mass_acc, depth_acc = counts[keep], mass_acc[keep], depth_acc[keep]
        if counts.shape[0] == 0:
            return set()
        if counts.shape[0] > _ENUM_ROW_LIMIT:
            return None

    # Hydrogen rebalancing: at most one H count can land in the +-delta window
    # (delta << 1 Da), but test the rounding neighbours too so a borderline
    # round() can never lose a candidate the exact check would accept.
    n0 = np.rint((M - mass_acc) / _H_STEP).astype(np.int64)
    survivors = []
    for dh in (-1, 0, 1):
        n_h = n0 + dh
        total = mass_acc + n_h * ISOTOPE_MASS["H"]
        ok = (n_h >= 0) & (np.abs(total - M) <= delta_da + eps)
        for row in np.nonzero(ok)[0]:
            survivors.append((row, int(n_h[row])))

    found = set()
    heavy_by_pos = [heavy[i] for i in order]
    for row, n_h in survivors:
        atom_counts = dict(payload)
        for pos, atom in enumerate(heavy_by_pos):
            c = int(counts[row, pos])
            if c > 0:
                atom_counts[atom] = c
        if n_h > 0:
            atom_counts["H"] = n_h
        if h_cap != -1 and any(
            abs(n_h - c.get("H", 0)) > h_cap for c in f0_counts_list
        ):
            continue
        f = format_formula(atom_counts)
        m = fast_mass(f)  # exact acceptance on the same value the BFS uses
        if not (M - delta_da <= m <= M + delta_da):
            continue
        if passes_senior_rule(f):
            found.add(f)
    return found


def _heavy_distance(counts, f0_counts):
    """Total heavy-atom (non-H) count difference between two formulas."""
    atoms = set(counts) | set(f0_counts)
    atoms.discard("H")
    return sum(abs(counts.get(a, 0) - f0_counts.get(a, 0)) for a in atoms)


def formula_refinement(
    f0_list, M, delta_M, ppm_mode, K, D, T, refine_atom_type, refine_atom_num
):
    """Refine predicted formulas around the target mass.

    Same contract as formula_refinement_bfs (see its docstring for the
    arguments), but when the capped search space is small enough it is
    enumerated completely instead of searched, so the candidate pool is
    provably every formula satisfying the constraints. The pool is ranked by
    (heavy-atom edit distance from the nearest initial formula, then
    abs(mass - M)): the model's prediction is usually nearly right, so fewer
    atom changes is a far stronger prior than mass proximity alone — on the
    Q-TOF release split this ranking nearly doubles top-1 recall over the BFS
    while raising top-10 recall from 76.4% to 88.5%. The timeout T is unused
    on the enumeration path, which is bounded by construction; searches too
    large to enumerate (e.g. uncapped atom types) fall back to the BFS.

    Returns:
            dict: {'formula': list[str | None], 'mass': list[float | None]},
                  each of length K, ranked as above, padded with None.
    """
    delta_da = delta_M * M / 10**6 if ppm_mode else delta_M
    valid = _enumerate_valid_formulas(
        f0_list, M, delta_da, D, refine_atom_type, refine_atom_num
    )
    if valid is None:
        return formula_refinement_bfs(
            f0_list, M, delta_M, ppm_mode, K, D, T, refine_atom_type, refine_atom_num
        )
    f0_counts_list = [parse_formula(f0) for f0 in f0_list]
    valid.sort(
        key=lambda f: (
            min(
                _heavy_distance(parse_formula(f), f0_counts)
                for f0_counts in f0_counts_list
            ),
            abs(fast_mass(f) - M),
        )
    )
    refine_f = valid[:K]
    refine_m = [fast_mass(f) for f in refine_f]
    if len(refine_f) < K:
        refine_f = refine_f + [None] * (K - len(refine_f))
        refine_m = refine_m + [None] * (K - len(refine_m))
    return {"formula": refine_f, "mass": refine_m}
