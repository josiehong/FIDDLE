Troubleshooting Low Accuracy
============================

When msfiddle finds the right formula less often than you expect, the cause is
usually the input or the evaluation rather than the model. Rule out the
installation first, then work through the checks below before retraining.

Rule out the installation
-------------------------

Run FIDDLE alone on the benchmarks, as :ref:`reproducing-benchmarks`
describes (``experiments_test_benchmark.sh --fiddle_only``, which needs
nothing but msfiddle). If its accuracy matches the FIDDLE bars of the
:doc:`benchmarks` figure, the installation and checkpoints are fine and the
cause lies in your data or evaluation.

Read the numbers first
----------------------

``--eval`` reports four things that are easy to mix up:

- **Top-1 and top-k.** Top-1 counts a spectrum as correct only when the
  highest-ranked candidate is the true formula; top-k when the true formula is
  anywhere among the first k. Top-k is cumulative, so it never falls as k
  grows.
- **Per spectrum and per compound.** Per-spectrum accuracy is the share of
  spectra. Per-compound accuracy averages each compound's per-spectrum
  accuracy, so a compound measured fifty times weighs as much as one measured
  once. When the two differ a lot, a few heavily replicated compounds dominate
  one of them.
- **Coverage.** Accuracy is over the result rows whose ``ID`` matches an MGF
  ``TITLE`` with a ``SMILES`` or ``FORMULA``. Spectra that ``predict`` skipped,
  and rows without a truth, are left out, not counted as wrong.

``--eval_details`` prints the coverage counts and saves each spectrum's rank of
the true formula and its metadata (see :doc:`usage`). Look at the coverage
first: a high accuracy on a small share of the file is a different problem from
a low accuracy on all of it.

Checks before retraining
------------------------

1. **Validate the input.** ``msfiddle validate`` checks every spectrum against
   the model's encoding without loading the models, and gives the reason for
   each one ``predict`` would skip:

   .. code-block:: bash

      msfiddle validate --test_data data.mgf --instrument_type orbitrap \
                        --report_path validation.csv

   If many spectra are rejected, fix the MGF before looking at accuracy. The
   required fields are listed in :doc:`formats`.

2. **Use the models of your instrument.** ``--instrument_type`` selects the
   models and config: ``orbitrap`` (the default), ``qtof`` or ``timstof``.
   Each pair of models was trained on spectra from that instrument type only,
   and the default is Orbitrap whatever the data.

3. **Check the collision energy convention.** A bare number such as ``35`` is
   read as eV. Orbitrap data usually report normalized collision energy, which
   must be written as NCE (``35 nce`` or ``NCE=35%``); otherwise the model is
   given an energy several times too high. Stepped energies
   (``HCD (NCE 20-30-40%)``) are skipped by the Orbitrap and Q-TOF models and
   averaged by the timsTOF model.

4. **Check the adducts.** Only the precursor types in :doc:`formats` are
   supported, and the adduct must be the one actually measured: a sodium
   adduct labelled ``[M+H]+`` gives a neutral mass 22 Da off, and no
   candidate can be right. ``--eval_details`` groups accuracy by adduct.

5. **Check the precursor mass error.** Refinement only considers formulas
   within ``post_processing.mass_tolerance`` of the measured precursor (5 ppm
   for Orbitrap, 10 ppm for Q-TOF and timsTOF). Data with larger errors, or
   precursors read from the M+1 isotope peak, need a config with a wider
   window or ``isotope_errors: [0, 1]``. ``scripts/check_precursor_error.py``
   in the repository classifies each spectrum's mass error as within
   tolerance, an isotope peak, or other.

6. **Check the evaluation truth.** The truth is the intact molecule from
   ``SMILES``, or ``FORMULA`` when there is no SMILES. A ``FORMULA`` field
   holding the ion's formula, or a salt or mixture SMILES, makes correct
   predictions look wrong. Titles must be unique: ``validate`` reports repeated
   ones, and only the last spectrum of a repeated title is predicted.

7. **Look at what the misses share.** ``<result>_eval_groups.csv`` gives
   accuracy by precursor m/z, adduct, peak count and NCE. When the misses
   gather in one group, such as one adduct or one energy range, check that
   group's metadata before anything else.

8. **Add BUDDY and SIRIUS candidates.** Ranking their candidates together
   with FIDDLE's often raises accuracy; :doc:`benchmarks` shows by how much,
   and :doc:`usage` how to run them.

Only when the input passes these checks and your data differ from the
training libraries (another instrument, an unusual compound class) is
retraining worth considering.

Reporting a problem
-------------------

If the accuracy is still low, open a
`low-accuracy issue <https://github.com/JosieHong/FIDDLE/issues/new?template=low_accuracy.yml>`_.
The form asks for the version, the command, the ``validate`` summary, the
coverage counts and the grouped accuracy, none of which contain your spectra.
