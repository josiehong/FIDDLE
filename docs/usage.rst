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

   # Download to the default location (~/.msfiddle/check_point)
   msfiddle-download-models

   # Download specific models to a custom location
   msfiddle-download-models --destination /path/to/models \
                             --models fiddle_tcn_qtof fiddle_rescore_qtof

To inspect current model paths:

.. code-block:: bash

   msfiddle-checkpoint-paths

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

``--instrument_type`` accepts ``orbitrap`` (default) or ``qtof``.

**Custom model paths:**

.. code-block:: bash

   msfiddle --test_data /path/to/data.mgf \
            --config_path /path/to/config.yml \
            --resume_path /path/to/tcn_model.pt \
            --rescore_resume_path /path/to/rescore_model.pt \
            --result_path /path/to/results.csv \
            --device 0

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
