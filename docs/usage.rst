Usage
=====

Installation
------------

.. code-block:: bash

   pip install "msfiddle[inference]"

The ``inference`` extra installs PyTorch, which prediction needs: a CUDA build
on Linux, a CPU build on macOS and Windows. For another build, such as another
CUDA version or CPU only on a cluster, install PyTorch first as the
`PyTorch guide <https://pytorch.org/get-started/locally/>`_ describes and then
run ``pip install msfiddle``; pip keeps the PyTorch that is already installed.
Without PyTorch, ``validate``, ``download`` and ``paths`` still work, and
``predict`` stops with an error that says what to install.

Downloading the models
----------------------

Model weights must be downloaded before running predictions:

.. code-block:: bash

   # Download to the default location (~/.msfiddle/check_point/<release tag>)
   msfiddle-download-models

   # Download one instrument's models to a custom location
   msfiddle-download-models --destination /path/to/models --instruments qtof

Each release keeps its checkpoints in its own directory (for 2.4.x,
``~/.msfiddle/check_point/v2.4.0``), so run ``msfiddle-download-models`` again
after upgrading to a new minor version.

To inspect current model paths:

.. code-block:: bash

   msfiddle-checkpoint-paths

Commands
--------

``msfiddle`` has four subcommands:

.. code-block:: text

   msfiddle predict   predict formulas for an MGF file
   msfiddle validate  check an MGF file's spectra without loading the models
   msfiddle download  download the pre-trained models
   msfiddle paths     show the checkpoint directory and downloaded models

Arguments without a subcommand go to ``predict``, so ``msfiddle --test_data ...``
still works. ``python -m msfiddle ...`` is equivalent, and also works from a
source checkout without installing the package. ``msfiddle-download-models``
and ``msfiddle-checkpoint-paths`` remain as aliases for ``download`` and
``paths``.

Checking the input
------------------

``validate`` checks each spectrum against an instrument's encoding, the same
checks ``predict`` runs, without loading or downloading the models:

.. code-block:: bash

   msfiddle validate --test_data /path/to/data.mgf \
                     --instrument_type orbitrap \
                     --report_path /path/to/validation.csv

It prints how many spectra are usable and counts the rejected ones by reason.
The report has one row per spectrum: ``index``, ``id``, ``status`` (``ok``,
``warning`` or ``rejected``), ``reason`` and ``message``. ``predict`` skips
exactly the rejected spectra. The reasons are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Reason
     - Rejected because
   * - ``missing_metadata``
     - ``TITLE``, ``PRECURSOR_MZ``, ``PRECURSOR_TYPE`` or ``COLLISION_ENERGY``
       is missing, or ``TITLE`` is empty
   * - ``duplicate_id``
     - a later usable spectrum has the same ``TITLE``; only the last usable
       one is predicted
   * - ``invalid_precursor_mz``
     - the precursor m/z is not a positive number
   * - ``unsupported_adduct``
     - the precursor type is not one the model encodes
   * - ``insufficient_peaks``
     - fewer than 5 peaks
   * - ``invalid_peaks``
     - non-finite or negative m/z values, or m/z and intensity values that do
       not match
   * - ``peak_mz_out_of_range``
     - the largest peak m/z is below 50 or above the encoder's ``max_mz``
   * - ``stepped_collision_energy``
     - a stepped energy, with a model that skips them
   * - ``invalid_collision_energy``
     - the collision energy cannot be parsed
   * - ``encoding_failure``
     - no peaks remain after removing the precursor isotope peaks, or all
       intensities are equal

Usable spectra can carry warnings: ``stepped_collision_energy_averaged``
(encoded at the mean energy, timsTOF only) and ``duplicate_id`` (the last usable
spectrum of a repeated ``TITLE``). ``validate`` exits with status 1 when the file cannot be
read or no spectrum is usable. ``msfiddle.validate_mgf`` returns the report as
a DataFrame.

Running predictions
-------------------

**Demo data:**

.. code-block:: bash

   msfiddle --demo --result_path ./output_demo.csv --device 0

**Custom data:**

.. code-block:: bash

   msfiddle --test_data /path/to/data.mgf \
            --instrument_type orbitrap \
            --result_path /path/to/results.csv \
            --device 0

``--instrument_type`` accepts ``orbitrap`` (default), ``qtof`` or ``timstof``.
The timsTOF models are trained on the Enveda CASMI 2026 library, and the
timsTOF config encodes spectra merged over several collision energies
(``20,40,60``) at their mean energy. For large
files, ``--num_workers N`` runs formula refinement in N processes (results are
identical) and ``--batch_size`` sets the inference batch (default 32).

**Custom model paths:**

.. code-block:: bash

   msfiddle --test_data /path/to/data.mgf \
            --config_path /path/to/config.yml \
            --resume_path /path/to/tcn_model.pt \
            --rescore_resume_path /path/to/rescore_model.pt \
            --result_path /path/to/results.csv \
            --device 0

A rescore model works only with the TCN it was trained on. Rescore
checkpoints record a hash of that TCN's weights, and prediction warns when
``--resume_path`` is a different TCN.

**Mass window and isotope errors:** two ``post_processing`` settings in the
config control which candidate masses refinement considers.

- ``mass_tolerance`` is the precursor mass error allowed, in ppm of the
  precursor m/z (5 for Orbitrap, 10 for Q-TOF and timsTOF). It is converted to the neutral
  mass of each adduct.
- ``isotope_errors`` lists 13C isotope offsets to try. With ``[0, 1]``,
  msfiddle also refines as if the precursor m/z were read from the M+1 isotope
  peak, and subtracts ``isotope_penalty`` (default 0.5) from the rescore of
  candidates found that way. The results then gain ``Isotope Shift (k)``
  columns. The default ``[0]`` searches the measured mass only.

Evaluating predictions
----------------------

If the true formulas are known, add ``--eval`` to ``predict``. After saving the
results it reports cumulative top-1 to top-k accuracy (k up to the number of
candidates), per spectrum and per compound with each compound weighted equally,
and saves the table as ``<result>_eval.csv``:

.. code-block:: bash

   msfiddle predict --test_data /path/to/data.mgf \
                    --result_path /path/to/results.csv \
                    --eval

The truth is each spectrum's ``SMILES`` field, or its ``FORMULA`` field when
there is no SMILES. Both describe the intact molecule, which is also what
msfiddle predicts for adducts with a neutral loss such as ``[M+H-H2O]+``.

``--eval_details`` (which implies ``--eval``) also prints the evaluation
coverage and saves two more files:

- ``<result>_eval_details.csv``: one row per result row, then one per spectrum
  with a truth but no result. ``eval_status`` is ``evaluated``, ``no_truth``
  or ``not_predicted``; only ``evaluated`` rows count towards the accuracy.
  Each row has the true formula, the top-1 prediction, ``true_rank`` (1-based,
  empty when the true formula is not among the returned candidates),
  cumulative ``top{k}_correct`` columns, and the spectrum's precursor m/z,
  adduct, collision energy and peak count. Averaging ``top{k}_correct`` over
  the evaluated rows, and over compounds, gives the numbers in
  ``<result>_eval.csv``.
- ``<result>_eval_groups.csv``: per-spectrum top-k accuracy of the evaluated
  rows by precursor m/z, adduct, peak count and NCE, with the spectrum and
  compound counts of each group. Stepped or unparseable energies fall in
  ``unknown``.

The coverage counts the input spectra, spectra with a truth, result rows,
evaluated rows, result rows without a truth, spectra with a truth but no
result row (usually rejected by ``validate``) and repeated IDs. Every result
row with a truth is evaluated, including repeated IDs. The result CSV holds only
the returned candidates, so an empty ``true_rank`` cannot tell a ranking miss
from a formula that refinement never generated.

:doc:`troubleshooting` describes what to check when the accuracy is low.

Integration with BUDDY and SIRIUS
---------------------------------

msfiddle can rank candidate formulas from the
`BUDDY/msbuddy command-line tool <https://msbuddy.readthedocs.io/en/latest/cmdapi.html>`_
and the `SIRIUS command-line interface <https://v6.docs.sirius-ms.io/cli/>`_
together with its own.
Neither tool comes with msfiddle; install them separately. The steps below run
both on ``data.mgf`` and compare FIDDLE alone, FIDDLE w. BUDDY and
FIDDLE w. BUDDY & SIRIUS.

**1. Prepare the input for the tools.** Both tools read the precursor m/z from
``PEPMASS``, the charge from ``CHARGE`` and the adduct from ``ION``. SIRIUS
needs the MS level as a number and reports its results under ``FEATURE_ID``,
so set that to the title msfiddle matches results on. SIRIUS also takes a ``FORMULA``, ``SMILES`` or
``INCHI`` field as the known answer, so remove them when the file holds
reference annotations, as test data does, and it reads any field starting with
``ION`` as the adduct, so a field like ``IONMODE=P`` must go too. This writes ``tools.mgf`` from an
msfiddle input file:

.. code-block:: python

   import re
   from pyteomics import mgf

   def for_tools(spectra):
       for spec in spectra:
           p = spec["params"]
           for key in list(p):
               if key in ("smiles", "inchi", "inchikey", "inchi_key", "formula") \
                       or key.startswith("ion"):
                   del p[key]
           p["feature_id"] = p["title"]
           p["pepmass"] = float(p["precursor_mz"])
           p["ion"] = p["precursor_type"]
           n, sign = re.search(r"\](\d*)([+-])$", p["ion"]).groups()
           p["charge"] = (n or "1") + sign
           p["ms_level"] = "2"
           yield spec

   mgf.write(for_tools(mgf.read("data.mgf")), "tools.mgf", file_mode="w")

**2. Run BUDDY.** Use ``-ms orbitrap`` or ``-ms qtof`` (also for timsTOF data):

.. code-block:: bash

   msbuddy -mgf tools.mgf -output buddy_output -ms orbitrap -d -p

``-d`` writes per-spectrum ``formula_results.tsv`` files with a score for
every candidate, which msfiddle uses; without it only the top candidate has a
score. msbuddy's models need ``lightgbm`` and ``scikit-learn``, which it does
not install itself.

**3. Run SIRIUS** with the matching profile and export the top five
candidates per spectrum:

.. code-block:: bash

   sirius --input tools.mgf --project sirius_project.sirius \
          formulas --profile orbitrap
   sirius --project sirius_project.sirius \
          summaries --top-k-summary=5 --output sirius_output

SIRIUS 6 requires ``sirius login`` before computing formulas. It skips spectra
it cannot read and still exits normally, so check that
``sirius_output/formula_identifications_top-5.tsv`` has rows. It adds to an
existing project, so delete ``sirius_project.sirius`` before running it on a
new file.

**4. Run FIDDLE** alone and with each tool's candidates. Give FIDDLE the
original file; with ``--eval`` it reads the true formulas from it:

.. code-block:: bash

   msfiddle --test_data data.mgf --instrument_type orbitrap \
            --result_path fiddle.csv --eval
   msfiddle --test_data data.mgf --instrument_type orbitrap \
            --buddy_path buddy_output \
            --result_path fiddle_buddy.csv --eval
   msfiddle --test_data data.mgf --instrument_type orbitrap \
            --buddy_path buddy_output --sirius_path sirius_output \
            --result_path fiddle_buddy_sirius.csv --eval

Each run writes its candidates to the result CSV and, with ``--eval``, the
top-k accuracy to ``<result>_eval.csv``. Leave out ``--eval`` for data without
known formulas, and either path to use one tool only.
``--buddy_path`` also accepts ``msbuddy_result_summary.tsv`` and
``--sirius_path`` a single formula summary file. See :doc:`formats` for these
formats and the deprecated msfiddle-normalized CSV files.

Python API
----------

For a single native MS/MS spectrum:

.. code-block:: python

   from msfiddle import predict_from_spectrum

   candidates = predict_from_spectrum(
       mz_array=[60.0, 85.0, 100.0, 125.0, 150.0],
       intensity_array=[10.0, 50.0, 20.0, 35.0, 15.0],
       precursor_mz=180.063,
       adduct="[M+H]+",
       top_k=5,
       instrument_type="orbitrap",
       collision_energy="Unknown",
       device="cpu",
   )

For repeated or batched use, instantiate a predictor once so model checkpoints
are loaded once and reused:

.. code-block:: python

   from msfiddle import MsFiddlePredictor

   predictor = MsFiddlePredictor(instrument_type="orbitrap", device="cpu")
   results = predictor.predict_batch(
       [
           {
               "id": "sample-1",
               "mz_array": [60.0, 85.0, 100.0, 125.0, 150.0],
               "intensity_array": [10.0, 50.0, 20.0, 35.0, 15.0],
               "precursor_mz": 180.063,
               "adduct": "[M+H]+",
               "collision_energy": "Unknown",
           }
       ]
   )

MGF files can also be used from Python:

.. code-block:: python

   from msfiddle import predict_from_mgf

   df = predict_from_mgf(
       "/path/to/data.mgf",
       instrument_type="orbitrap",
       device="cpu",
   )

The Python APIs are quiet by default and do not download checkpoints unless
``download_models=True`` is passed. The CLI also requires checkpoints to be
downloaded before prediction and prints a checkpoint error with the
``msfiddle-download-models`` command if they are missing.
