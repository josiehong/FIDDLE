Benchmarks
==========

Formula accuracy of the released models on three public benchmarks, for
msfiddle |release|. Top-k is the share of spectra whose true formula is among
the k highest-ranked candidates.

The benchmarks are Q-TOF data and use the Q-TOF models with the benchmark
configs ``fiddle_tcn_casmi.yml`` and ``fiddle_tcn_embl.yml``. Each is run
three ways:

- **FIDDLE**: FIDDLE's own candidates.
- **FIDDLE + BUDDY**: BUDDY's candidates added to FIDDLE's, all ranked by the
  FIDDLE rescore model.
- **FIDDLE + BUDDY + SIRIUS**: SIRIUS's candidates added as well.

Some benchmark compounds are also in the training libraries, so each result
is given on all compounds and on the unseen ones, those absent from the
models' training and validation data. Compounds are matched on the first
block of the InChIKey, so a stereoisomer of a training compound counts as
seen.

.. list-table:: CASMI 2016
   :header-rows: 1

   * - Method
     - Compounds
     - Top-1
     - Top-3
     - Top-5
   * - FIDDLE
     - All
     - --
     - --
     - --
   * - FIDDLE
     - Unseen
     - --
     - --
     - --
   * - FIDDLE + BUDDY
     - All
     - --
     - --
     - --
   * - FIDDLE + BUDDY
     - Unseen
     - --
     - --
     - --
   * - FIDDLE + BUDDY + SIRIUS
     - All
     - --
     - --
     - --
   * - FIDDLE + BUDDY + SIRIUS
     - Unseen
     - --
     - --
     - --

.. list-table:: CASMI 2017
   :header-rows: 1

   * - Method
     - Compounds
     - Top-1
     - Top-3
     - Top-5
   * - FIDDLE
     - All
     - --
     - --
     - --
   * - FIDDLE
     - Unseen
     - --
     - --
     - --
   * - FIDDLE + BUDDY
     - All
     - --
     - --
     - --
   * - FIDDLE + BUDDY
     - Unseen
     - --
     - --
     - --
   * - FIDDLE + BUDDY + SIRIUS
     - All
     - --
     - --
     - --
   * - FIDDLE + BUDDY + SIRIUS
     - Unseen
     - --
     - --
     - --

.. list-table:: EMBL-MCF 2.0
   :header-rows: 1

   * - Method
     - Compounds
     - Top-1
     - Top-3
     - Top-5
   * - FIDDLE
     - All
     - --
     - --
     - --
   * - FIDDLE
     - Unseen
     - --
     - --
     - --
   * - FIDDLE + BUDDY
     - All
     - --
     - --
     - --
   * - FIDDLE + BUDDY
     - Unseen
     - --
     - --
     - --
   * - FIDDLE + BUDDY + SIRIUS
     - All
     - --
     - --
     - --
   * - FIDDLE + BUDDY + SIRIUS
     - Unseen
     - --
     - --
     - --

Reproducing the results
-----------------------

From a clone of the repository, with BUDDY (``msbuddy``) and SIRIUS on the
``PATH`` and the raw benchmark files in ``data/benchmark/``:

.. code-block:: bash

   msfiddle download
   bash running_scripts/experiments_test_benchmark.sh

The raw files are the CASMI 2016 and EMBL-MCF 2.0 HRMS library exports from
`MoNA <https://mona.fiehnlab.ucdavis.edu>`_ and the CASMI 2017 challenge
files and solutions from the `CASMI contest <http://www.casmi-contest.org>`_:

.. code-block:: text

   data/benchmark/
   ├── casmi2016/MoNA-export-CASMI_2016.sdf
   ├── casmi2017/
   │   ├── Chal1to45Summary.csv
   │   ├── CASMI-solutions.csv
   │   ├── challenges-001-045-msms-mgf-20170908/
   │   └── challenges-001-045-ms-mgf-20170908/
   └── embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf

The script writes top-1 to top-20 accuracy, per spectrum and per compound, to
``result/<method>_<benchmark>_eval.csv``, where the method is ``fiddle``,
``fiddle_buddy`` or ``fiddle_buddy_sirius`` and the benchmark ``casmi2016``,
``casmi2017`` or ``embl``, with an ``_unseen`` suffix for the unseen compounds.
``scripts/remove_training_compounds.py`` builds the unseen sets from the
training data the models were built on (``DATA_TAG``), so that data must be in
``data/cl_pkl_<DATA_TAG>/``.

The `paper <https://www.nature.com/articles/s41467-025-66060-9>`_ reports the
full evaluation, including held-out test sets and comparisons with other
tools.
