Troubleshooting low accuracy
============================

When msfiddle finds the right formula less often than you expect, the cause is
usually the input or the evaluation, not the model. Work through these checks
before retraining.

Rule out the installation
-------------------------

Run ``bash running_scripts/experiments_test_benchmark.sh --fiddle_only``, as
:ref:`reproducing-benchmarks` describes. If it matches the FIDDLE bars of the
:doc:`benchmarks` figure, the installation and checkpoints are fine.

Check what the numbers count
----------------------------

Top-k counts a spectrum as correct when the true formula is among the first k
candidates, and per-compound accuracy weighs every compound equally, however
many spectra it has. Spectra that ``predict`` skipped, and rows without a true
formula, are left out of the accuracy rather than counted as wrong. So look at
the coverage that ``--eval_details`` prints first: a high accuracy on a small
share of the file is a different problem from a low accuracy on all of it.

Check the input
---------------

1. Run ``msfiddle validate`` (see :doc:`usage`). It gives the reason for every
   spectrum ``predict`` would skip; fix those first.
2. Set ``--instrument_type`` to your instrument. It defaults to ``orbitrap``
   whatever the data.
3. Write normalized collision energies as NCE (``35 nce``). A bare number is
   read as eV, which gives the model an energy several times too high.
4. Make sure ``PRECURSOR_TYPE`` is the adduct actually measured. A sodium
   adduct labelled ``[M+H]+`` puts the neutral mass 22 Da off, and then no
   candidate can be right.
5. Compare your precursor mass errors with the search window: 5 ppm for
   Orbitrap, 10 ppm for Q-TOF and timsTOF. ``scripts/check_precursor_error.py``
   in the repository measures them. For larger errors, or precursors read from
   the M+1 isotope peak, set a wider ``post_processing.mass_tolerance`` or
   ``isotope_errors: [0, 1]`` in a copy of the config.
6. Check the truth: the intact molecule, from ``SMILES`` or else ``FORMULA``.
   An ion formula in ``FORMULA``, or the SMILES of a salt or mixture, makes
   correct predictions look wrong.

``<result>_eval_groups.csv`` gives the accuracy by precursor m/z, adduct, peak
count and NCE. If the misses gather in one group, start with that group's
metadata. Adding BUDDY's and SIRIUS's candidates often raises accuracy further
(see :doc:`benchmarks`).

Retrain only when the input passes these checks and your data differ from the
training libraries, for example another instrument or an unusual compound
class.

Reporting a problem
-------------------

If the accuracy is still low, open a
`low-accuracy issue <https://github.com/JosieHong/FIDDLE/issues/new?template=low_accuracy.yml>`_.
It asks for the version, the command, the ``validate`` summary, the coverage
and the grouped accuracy, none of which contain your spectra.
