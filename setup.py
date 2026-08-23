from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="msfiddle",
    version="2.1.0",
    author="Yuhui Hong",
    author_email="josieexception@outlook.com",
    description="A package for predicting chemical formulas from tandem mass spectra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JosieHong/FIDDLE",
    packages=find_packages(include=["msfiddle", "msfiddle.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
        "pandas>=2.0.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.8.0",
        "pyarrow>=10.0.0",
        "rdkit>=2022.03.5",
        "molmass",
        "pyteomics",
    ],
    extras_require={
        "inference": ["torch"],
    },
    entry_points={
        "console_scripts": [
            "msfiddle=msfiddle.main:main",
            "msfiddle-download-models=msfiddle.download:download_models_cli",
            "msfiddle-checkpoint-paths=msfiddle.download:print_checkpoint_paths_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "msfiddle": [
            "config/*.yml",
            "demo/*.mgf",
            "demo/*.csv",
        ],
    },
)
