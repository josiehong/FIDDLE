Usage
=====

Installation
------------

.. code-block:: bash

   pip install msfiddle

PyTorch must be installed separately following the
`official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.
Alternatively, install the optional inference extra:

.. code-block:: bash

   pip install "msfiddle[inference]"

Downloading Pre-trained Models
-------------------------------

Model weights must be downloaded before running predictions:

.. code-block:: bash

   # Download to the default location (~/.msfiddle/check_point/<release tag>)
   msfiddle-download-models

   # Download specific models to a custom location
   msfiddle-download-models --destination /path/to/models \
                             --models fiddle_tcn_qtof fiddle_rescore_qtof

Each release keeps its checkpoints in its own directory (for 2.2.x,
``~/.msfiddle/check_point/v2.2.0``), so run ``msfiddle-download-models`` again
after upgrading to a new minor version.

To inspect current model paths:

.. code-block:: bash

   msfiddle-checkpoint-paths

Commands
--------

``msfiddle`` has three subcommands:

.. code-block:: text

   msfiddle predict   predict formulas for an MGF file
   msfiddle download  download the pre-trained models
   msfiddle paths     show the checkpoint directory and downloaded models

Arguments without a subcommand go to ``predict``, so ``msfiddle --test_data ...``
still works. ``python -m msfiddle ...`` is equivalent, and also works from a
source checkout without installing the package. ``msfiddle-download-models``
and ``msfiddle-checkpoint-paths`` remain as aliases for ``download`` and
``paths``.

Running Predictions
--------------------

**Demo data:**

.. code-block:: bash

   msfiddle --demo --result_path ./output_demo.csv --device 0

**Custom data:**

.. code-block:: bash

   msfiddle --test_data /path/to/data.mgf \
            --instrument_type orbitrap \
            --result_path /path/to/results.csv \
            --device 0

``--instrument_type`` accepts ``orbitrap`` (default) or ``qtof``. For large
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

**Mass window and isotope errors:** two ``post_processing`` settings in the
config control which candidate masses refinement considers.

- ``mass_tolerance`` is the precursor mass error allowed, in ppm of the
  precursor m/z (5 for Orbitrap, 10 for Q-TOF). It is converted to the neutral
  mass of each adduct.
- ``isotope_errors`` lists 13C isotope offsets to try. With ``[0, 1]``,
  msfiddle also refines as if the precursor m/z were read from the M+1 isotope
  peak, and subtracts ``isotope_penalty`` (default 0.5) from the rescore of
  candidates found that way. The results then gain ``Isotope Shift (k)``
  columns. The default ``[0]`` searches the measured mass only.

Evaluating Predictions
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

Integration with BUDDY and SIRIUS
-----------------------------------

Candidate formulas from the `BUDDY/msbuddy command-line tool
<https://msbuddy.readthedocs.io/en/latest/cmdapi.html>`_ and the
`SIRIUS command-line interface <https://v6.docs.sirius-ms.io/cli/>`_ can be
incorporated to improve refinement results. ``--buddy_path`` and
``--sirius_path`` accept native/original tool outputs. The older
msfiddle-normalized CSV files are still accepted but are deprecated and will be
removed in ``msfiddle`` 3.0.0.

First, run BUDDY/msbuddy with the same MGF file:

.. code-block:: bash

   msbuddy -mgf /path/to/data.mgf \
           -output /path/to/buddy_output \
           -ms orbitrap \
           -d

``msbuddy`` writes ``msbuddy_result_summary.tsv`` in the output directory.
When ``-d`` is used, it also writes per-spectrum ``formula_results.tsv`` files
with per-candidate FDR scores. Passing the full output directory to
``msfiddle`` is preferred because those detailed scores can be used directly.
If only ``msbuddy_result_summary.tsv`` is passed, only rank 1 has an FDR score
and lower ranks are not used by the FDR threshold. See the
`msbuddy command-line API <https://msbuddy.readthedocs.io/en/latest/cmdapi.html>`_
for the full option list.

Next, run SIRIUS and export formula summaries:

.. code-block:: bash

   sirius --input /path/to/data.mgf \
          --project /path/to/sirius_project \
          formulas --profile orbitrap

   sirius --project /path/to/sirius_project \
          summaries --top-k-summary=5 \
          --output /path/to/sirius_output

SIRIUS writes formula summary files such as
``formula_identifications.tsv`` or ``formula_identifications_top-5.tsv``.
SIRIUS 6 may require ``sirius login`` before formula computation. See the
`SIRIUS command-line interface <https://v6.docs.sirius-ms.io/cli/>`_ for
workflow details and the full option list.

Then pass those native/original outputs to ``msfiddle``:

.. code-block:: bash

   msfiddle --test_data /path/to/data.mgf \
            --buddy_path /path/to/buddy_output \
            --sirius_path /path/to/sirius_output \
            --result_path /path/to/results.csv \
            --device 0

You can also pass ``/path/to/buddy_output/msbuddy_result_summary.tsv`` or an
individual SIRIUS formula summary file directly. If only one external tool is
available, omit the other option. See :doc:`formats` for the native formats
and the deprecated msfiddle-normalized schemas.

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
