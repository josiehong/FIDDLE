File Formats
============

Input: MGF
----------

msfiddle accepts tandem mass spectra in ``.mgf`` format. Four fields are
required per spectrum; all others are ignored.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``TITLE``
     - Unique spectrum identifier, propagated to the ``ID`` column in output.
   * - ``PRECURSOR_MZ``
     - Observed precursor m/z.
   * - ``PRECURSOR_TYPE``
     - Adduct type (e.g. ``[M+H]+``, ``[M-H]-``). See supported types below.
   * - ``COLLISION_ENERGY``
     - Collision energy in eV.

**Example:**

.. code-block:: text

   BEGIN IONS
   TITLE=EMBL_MCF_2_0_HRMS_Library000529
   PEPMASS=111.02016
   CHARGE=1-
   PRECURSOR_TYPE=[M-H]-
   PRECURSOR_MZ=111.02016
   COLLISION_ENERGY=50.0
   41.0148 0.329893
   68.0258 0.402906
   111.0203 100.0
   END IONS

**Supported precursor types:**

``[M+H]+``, ``[M+2H]2+``, ``[M+Na]+``, ``[M-H]-``, ``[M+H-H2O]+``,
``[M-H2O+H]+``, ``[2M+H]+``, ``[2M-H]-``, ``[M+H-2H2O]+``, ``[M+H-NH3]+``,
``[M+H+NH3]+``, ``[M+NH4]+``, ``[M+H-CH2O2]+``, ``[M+H-CH4O2]+``,
``[M-H-CO2]-``, ``[M-CHO2]-``, ``[M-H-H2O]-``

Output: msfiddle CSV
--------------------

One row is produced per input spectrum.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``ID``
     - Spectrum identifier from the MGF ``TITLE`` field.
   * - ``Y Pred``
     - Raw TCN output: semicolon-separated atom-count vector
       (C, H, O, N, F, S, Cl, P, B, Br, I, Na, K).
   * - ``Mass``
     - Neutral monoisotopic mass derived from ``PRECURSOR_MZ`` and
       ``PRECURSOR_TYPE``.
   * - ``Pred Formula``
     - Top formula from the TCN model prior to refinement.
   * - ``Pred Mass``
     - Monoisotopic mass of ``Pred Formula``.
   * - ``Pred Atom Num``
     - Total atom count predicted by the model.
   * - ``Pred H/C Num``
     - H/C ratio predicted by the model.
   * - ``Running Time``
     - Wall time per spectrum in seconds (prediction + refinement).
   * - ``Refined Formula (k)``
     - The k-th best refined formula (0-indexed), ranked by rescore score.
       ``None`` if fewer than k+1 candidates were found.
   * - ``Refined Mass (k)``
     - Monoisotopic mass of ``Refined Formula (k)``.
   * - ``Rescore (k)``
     - Rescore model confidence score for ``Refined Formula (k)``
       (0–1; higher is more confident).

The number of ranked columns is set by ``top_k`` in the configuration file
(default: 5).

Native/Original BUDDY and SIRIUS Outputs
----------------------------------------

The commands below use the same MGF input file used by ``msfiddle``. They
produce native BUDDY/msbuddy and SIRIUS outputs that can be passed directly to
``msfiddle`` with ``--buddy_path`` or ``--sirius_path``.

BUDDY / msbuddy
~~~~~~~~~~~~~~~

BUDDY is available through the `msbuddy command-line tool
<https://msbuddy.readthedocs.io/en/latest/cmdapi.html>`_. It writes a
result-summary TSV file to the output directory; ``-d`` also writes detailed
per-query candidate results.

.. code-block:: bash

   msbuddy -mgf /path/to/data.mgf \
           -output /path/to/buddy_output \
           -ms orbitrap \
           -p -n_cpu 12 \
           -d -hal

Use ``-ms qtof`` or ``-ms fticr`` instead of ``-ms orbitrap`` when appropriate.
Native summary fields include ``identifier``, ``mz``, ``rt``, ``adduct``,
``formula_rank_1`` through ``formula_rank_5``, and ``estimated_fdr``. Keep the
MGF ``TITLE`` values aligned with the identifiers reported by msbuddy.

SIRIUS
~~~~~~

The `SIRIUS command-line interface <https://v6.docs.sirius-ms.io/cli/>`_ first
writes results to a project space. The ``summaries`` / ``write-summaries``
step exports tabular `summary files <https://v6.docs.sirius-ms.io/io/>`_ from
that project space.

.. code-block:: bash

   sirius --input /path/to/data.mgf \
          --project /path/to/sirius_project \
          formulas --profile orbitrap

   sirius --project /path/to/sirius_project \
          summaries --top-k-summary=5 \
          --output /path/to/sirius_output

Use ``--profile qtof`` instead of ``--profile orbitrap`` for Q-TOF data. The
molecular formula summary is exported as ``formula_identifications.tsv`` by
default, or as ``formula_identifications.[tsv|csv|xlsx]`` depending on the
selected export format. SIRIUS summary columns include formula/adduct
identifiers and scores such as ``molecularFormula``, ``adduct``, ``rank`` or
``rankingScore``, ``SiriusScore``, ``TreeScore``, ``IsotopeScore``,
mass-error fields, and explained-peak fields. SIRIUS 6 may require
``sirius login`` before formula computation.

If the project space already exists, run only the ``summaries`` command. The
resulting top-k or all-hit SIRIUS summaries can be passed to ``--sirius_path``
directly.

Optional BUDDY/msbuddy Results
------------------------------

``--buddy_path`` accepts:

* a native/original msbuddy output directory containing
  ``msbuddy_result_summary.tsv``;
* the native/original ``msbuddy_result_summary.tsv`` file itself; or
* an msfiddle-normalized CSV with the columns below (**deprecated**).

When a native/original msbuddy output directory includes detailed
``formula_results.tsv`` files from the ``-d`` option, msfiddle uses their
per-candidate FDR scores. Passing the full output directory is preferred.
When only the summary TSV is available, msfiddle can use the top-ranked formula
and its ``estimated_fdr`` score; lower ranks are retained without scores and
are therefore ignored by the configured FDR threshold.

msfiddle-normalized BUDDY CSV schema (deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. deprecated:: 2.1.0
   Use native msbuddy output instead. The msfiddle-normalized BUDDY CSV format
   will be removed in ``msfiddle`` 3.0.0; loading it emits a
   ``DeprecationWarning``.

Required columns:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``ID``
     - Spectrum identifier matching the MGF ``TITLE`` field.
   * - ``Adduct``
     - Precursor type string.
   * - ``Pred Formula (1–5)``
     - Top-5 candidate formulas from BUDDY.
   * - ``BUDDY Score (1–5)``
     - Confidence scores; candidates below the configured threshold are excluded.

Optional SIRIUS Results
-----------------------

``--sirius_path`` accepts:

* a SIRIUS summary output directory containing ``formula_identifications``
  files;
* a native/original SIRIUS ``formula_identifications`` summary file in TSV,
  CSV, or XLSX form; or
* an msfiddle-normalized CSV with the columns below (**deprecated**).

For native SIRIUS summaries, msfiddle reads identifier, molecular formula,
adduct, rank, and score columns from the formula-identification output, then
keeps the top five candidates per spectrum.

msfiddle-normalized SIRIUS CSV schema (deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. deprecated:: 2.1.0
   Use native SIRIUS output instead. The msfiddle-normalized SIRIUS CSV format
   will be removed in ``msfiddle`` 3.0.0; loading it emits a
   ``DeprecationWarning``.

Required columns:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``ID``
     - Spectrum identifier matching the MGF ``TITLE`` field.
   * - ``Pred Formula (1–5)``
     - Top-5 candidate formulas from SIRIUS.
   * - ``Pred Adduct (1–5)``
     - Predicted adduct for each candidate.
   * - ``SIRIUS Score (1–5)``
     - Log-likelihood scores; candidates below the configured threshold
       are excluded.
