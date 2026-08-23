import os
import sys
from importlib.metadata import version as get_version, PackageNotFoundError
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

project = "msfiddle"
author = "Yuhui Hong"
copyright = f"{date.today().year}, Yuhui Hong"
try:
    release = get_version("msfiddle")
except PackageNotFoundError:
    release = "unknown"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autodoc_member_order = "bysource"
autodoc_mock_imports = ["torch", "tqdm", "yaml", "pandas", "sklearn", "scipy"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "rdkit": ("https://www.rdkit.org/docs", None),
}

html_theme = "sphinx_rtd_theme"
html_title = "msfiddle"
