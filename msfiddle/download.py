import os
import sys
import shutil
import tempfile
import urllib.request
import zipfile
import argparse
import re
from pathlib import Path
from tqdm import tqdm

from importlib.metadata import (
    version as _get_version,
    PackageNotFoundError as _PackageNotFoundError,
)


def checkpoint_release_for_package_version(package_version):
    """Return the major-version FIDDLE release tag that contains checkpoints."""
    match = re.match(r"^(\d+)", str(package_version))
    if match is None:
        return f"v{package_version}"
    return f"v{match.group(1)}.0.0"


try:
    FIDDLE_RELEASE = checkpoint_release_for_package_version(_get_version("msfiddle"))
except _PackageNotFoundError:
    FIDDLE_RELEASE = "v0.0.0"

MODEL_URLS = {
    "fiddle_tcn_qtof": f"https://github.com/JosieHong/FIDDLE/releases/download/{FIDDLE_RELEASE}/fiddle_tcn_qtof.zip",
    "fiddle_rescore_qtof": f"https://github.com/JosieHong/FIDDLE/releases/download/{FIDDLE_RELEASE}/fiddle_rescore_qtof.zip",
    "fiddle_tcn_orbitrap": f"https://github.com/JosieHong/FIDDLE/releases/download/{FIDDLE_RELEASE}/fiddle_tcn_orbitrap.zip",
    "fiddle_rescore_orbitrap": f"https://github.com/JosieHong/FIDDLE/releases/download/{FIDDLE_RELEASE}/fiddle_rescore_orbitrap.zip",
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads using tqdm."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file with a progress bar."""
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def get_checkpoint_dir():
    """Get the directory where model checkpoints are stored.

    Returns ``~/.msfiddle/check_point``, a user-level directory that is
    consistent regardless of working directory or install method.
    """
    return os.path.join(os.path.expanduser("~"), ".msfiddle", "check_point")


def download_models(destination=None, models=None):
    """
    Download pre-trained models for msfiddle.

    Args:
        destination (str, optional): Path to save the models. If None, uses the package directory.
        models (list, optional): List of model names to download. If None, downloads all models.
    """
    if destination is None:
        destination = get_checkpoint_dir()
    else:
        destination = os.path.abspath(destination)

    # Create the destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)

    print(f"Models will be downloaded to: {destination}")

    # Determine which models to download
    if models is None:
        models = list(MODEL_URLS.keys())

    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        for model_name in models:
            if model_name not in MODEL_URLS:
                print(f"Warning: Unknown model '{model_name}', skipping.")
                continue

            url = MODEL_URLS[model_name]
            zip_path = os.path.join(temp_dir, f"{model_name}.zip")

            # Download the model
            print(f"Downloading {model_name} model...")
            try:
                download_url(url, zip_path)
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
                print(f"You can manually download it from {url}")
                continue

            # Extract the model
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Move the model file to the destination
                model_file = f"{model_name}.pt"
                src_path = os.path.join(temp_dir, model_file)
                dest_path = os.path.join(destination, model_file)

                if os.path.exists(src_path):
                    shutil.move(src_path, dest_path)
                    print(f"Installed {model_file} to {dest_path}")
                else:
                    print(
                        f"Warning: Model file {model_file} not found after extraction"
                    )
            except Exception as e:
                print(f"Error extracting {model_name}: {e}")
                print(f"You can manually download and extract it from {url}")

    print("\nDownload complete!")
    print(f"Models are installed in: {destination}")


def check_models_exist():
    """Check if models exist in the package directory."""
    model_dir = get_checkpoint_dir()

    if not os.path.exists(model_dir):
        return False

    # Check if all model files exist
    all_exist = True
    for model_name in MODEL_URLS.keys():
        model_path = os.path.join(model_dir, f"{model_name}.pt")
        if not os.path.exists(model_path):
            all_exist = False
            break

    return all_exist


def get_model_path(model_name):
    """Get the path to a specific model file."""
    model_dir = get_checkpoint_dir()
    return os.path.join(model_dir, f"{model_name}.pt")


def print_checkpoint_paths():
    """Print the checkpoint directory path and available models."""
    checkpoint_dir = get_checkpoint_dir()
    print(f"Checkpoint directory: {checkpoint_dir}")

    if os.path.exists(checkpoint_dir):
        print("\nAvailable models:")
        for model_name in MODEL_URLS.keys():
            model_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
            if os.path.exists(model_path):
                print(f"  [+] {model_name}.pt")
            else:
                print(f"  [-] {model_name}.pt (not downloaded)")
    else:
        print("\nCheckpoint directory does not exist. No models are downloaded.")
        print("Run 'msfiddle-download-models' to download the models.")


# Command-line interface for the download_models function
def download_models_cli():
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for msfiddle"
    )
    parser.add_argument(
        "-d",
        "--destination",
        help="Path to save the models (optional, defaults to package directory)",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        help="Specific model names to download (optional, defaults to all models)",
    )
    args = parser.parse_args()

    # If destination is not provided, it will be None and download_models will use get_checkpoint_dir()
    download_models(destination=args.destination, models=args.models)


# Command-line interface for the print_checkpoint_paths function
def print_checkpoint_paths_cli():
    parser = argparse.ArgumentParser(
        description="Print checkpoint paths and available models"
    )
    parser.parse_args()  # This just consumes any arguments passed
    print_checkpoint_paths()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download.py [command]")
        print("Commands:")
        print(
            "  print_checkpoint_paths - Print the checkpoint directory path and available models"
        )
        print(
            "  download_models [destination] - Download pre-trained models to the specified destination"
        )
        sys.exit(1)

    if sys.argv[1] == "print_checkpoint_paths":
        print_checkpoint_paths()

    elif sys.argv[1] == "download_models":
        # If the first argument is 'download', treat the second as destination
        if len(sys.argv) > 2:
            download_models(destination=sys.argv[2])
        else:
            download_models()

    else:
        raise ValueError(f"Unknown command: {sys.argv[1]}")
