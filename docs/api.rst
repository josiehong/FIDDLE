API Reference
=============

Prediction API
--------------

.. autoclass:: msfiddle.api.MsFiddlePredictor
   :members:

.. autofunction:: msfiddle.api.predict_from_spectrum

.. autofunction:: msfiddle.api.predict_batch_from_spectra

.. autofunction:: msfiddle.api.predict_from_mgf

Molecular Utilities
-------------------

.. automodule:: msfiddle.utils.mol_utils
   :members:
   :show-inheritance:
   :exclude-members: ATOMS_WEIGHT, ATOMS_VALENCE, ATOMS_INDEX, ATOMS_INDEX_re

MS/MS Utilities
---------------

.. automodule:: msfiddle.utils.msms_utils
   :members:
   :show-inheritance:
   :exclude-members: MSLEVEL_MAP, mgf_key_order

Formula Refinement
------------------

.. automodule:: msfiddle.utils.refine_utils
   :members:
   :show-inheritance:
