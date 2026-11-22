File formats
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
       Of spectra sharing a title, only the last is predicted.
   * - ``PRECURSOR_MZ``
     - Observed precursor m/z.
   * - ``PRECURSOR_TYPE``
     - Adduct type (e.g. ``[M+H]+``, ``[M-H]-``). See supported types below.
   * - ``COLLISION_ENERGY``
     - A single collision energy, in eV (``50.0``) or as NCE (``35 nce``,
       ``NCE=35%``). A bare number is read as eV.
       Spectra with a stepped or ranged energy (``HCD (NCE 20-30-40%)``,
       ``30-50 eV``) are skipped by the Orbitrap and Q-TOF models, which were
       trained on single-energy spectra, and encoded at their mean energy by
       the timsTOF model.

**Example:**

.. code-block:: text

   BEGIN IONS
   TITLE=EMBL_MCF_2_0_HRMS_Library000529
   PEPMASS=111.02016
   CHARGE=1-
   PRECURSOR_TYPE=[M-H]-
   PRECURSOR_MZ=111.02016
   COLLISION_ENERGY=50 nce
   41.0148 0.329893
   68.0258 0.402906
   111.0203 100.0
   END IONS

**Supported precursor types:**

``[M+H]+``, ``[M+2H]2+``, ``[M+Na]+``, ``[M-H]-``, ``[M+H-H2O]+``,
``[M-H2O+H]+``, ``[2M+H]+``, ``[2M-H]-``, ``[M+H-2H2O]+``, ``[M+H-NH3]+``,
``[M+H+NH3]+``, ``[M+NH4]+``, ``[M+H-CH2O2]+``, ``[M+H-CH4O2]+``,
``[M-H-CO2]-``, ``[M-CHO2]-``, ``[M-H-H2O]-``, ``[M-H2O-H]-``

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
   * - ``Isotope Shift (k)``
     - Only when the config's ``isotope_errors`` is not ``[0]``: the 13C
       isotope offset at which ``Refined Formula (k)`` was found (0 for the
       measured mass). The ranking uses ``Rescore (k)`` minus
       ``isotope_penalty`` times this offset.

The number of ranked columns is set by ``top_k`` in the configuration file
(20 in the bundled configs) unless ``top_k`` is passed
explicitly.

BUDDY and SIRIUS results
------------------------

:doc:`usage` shows how to run BUDDY and SIRIUS. msfiddle reads their native
output, so pass it as it is; the formats below are what it accepts.

BUDDY
~~~~~

``--buddy_path`` accepts:

* an msbuddy output directory containing ``msbuddy_result_summary.tsv``;
* the ``msbuddy_result_summary.tsv`` file itself; or
* an msfiddle-normalized CSV with the columns below (**deprecated**).

Pass the output directory where you can. When it holds the per-spectrum
``formula_results.tsv`` files that msbuddy's ``-d`` option writes, msfiddle
uses each candidate's FDR score. The summary TSV scores only the top-ranked
formula (``estimated_fdr``), so the configured FDR threshold drops the other
ranks.

Deprecated msfiddle-normalized BUDDY CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

SIRIUS
~~~~~~

``--sirius_path`` accepts:

* a directory written by ``sirius summaries``, containing
  ``formula_identifications`` files;
* one ``formula_identifications`` file, as TSV, CSV or XLSX (which needs
  ``pip install openpyxl``); or
* an msfiddle-normalized CSV with the columns below (**deprecated**).

msfiddle reads the identifier, molecular formula, adduct, rank and score
columns, and keeps the top five candidates per spectrum.

Deprecated msfiddle-normalized SIRIUS CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
