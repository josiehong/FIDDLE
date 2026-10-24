import os
import sys
import shutil
import tempfile
import urllib.request
import zipfile
import argparse
import re
from tqdm import tqdm

from importlib.metadata import (
    version as _get_version,
    PackageNotFoundError as _PackageNotFoundError,
)


def checkpoint_release_for_package_version(package_version):
    """Return the FIDDLE release tag that holds this version's checkpoints.

    From 2.2 on, each minor version has its own checkpoints (tag vX.Y.0,
    shared by its patch releases); 2.0 and 2.1 share the v2.0.0 assets.
    """
    match = re.match(r"^(\d+)(?:\.(\d+))?", str(package_version))
    if match is None:
        return f"v{package_version}"
    major, minor = int(match.group(1)), int(match.group(2) or 0)
    if (major, minor) < (2, 2):
        return f"v{major}.0.0"
    return f"v{major}.{minor}.0"


try:
    FIDDLE_RELEASE = checkpoint_release_for_package_version(_get_version("msfiddle"))
except _PackageNotFoundError:
    FIDDLE_RELEASE = "v0.0.0"

INSTRUMENTS = ("qtof", "orbitrap", "timstof")


def model_files(instrument):
    """The TCN and rescore checkpoint file names of an instrument."""
    return [f"fiddle_tcn_{instrument}.pt", f"fiddle_rescore_{instrument}.pt"]


def release_asset_urls(instrument, release=None):
    """URLs of the zips holding an instrument's checkpoints. From v2.3.0 on, one
    zip holds both, so the pair that belongs together is always installed
    together; earlier releases have a zip per model."""
    release = release or FIDDLE_RELEASE
    base = f"https://github.com/JosieHong/FIDDLE/releases/download/{release}"
    version = tuple(int(n) for n in re.findall(r"\d+", release)[:2])
    if version >= (2, 3):
        return [f"{base}/fiddle_{instrument}.zip"]
    return [f"{base}/{name[:-3]}.zip" for name in model_files(instrument)]


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

    Returns ``~/.msfiddle/check_point/<release tag>``, a user-level directory
    that is consistent regardless of working directory or install method. The
    tag keeps checkpoints from different releases apart, so an upgrade never
    silently reuses an older release's weights.
    """
    return os.path.join(
        os.path.expanduser("~"), ".msfiddle", "check_point", FIDDLE_RELEASE
    )


def download_models(destination=None, instruments=None):
    """
    Download pre-trained models for msfiddle.

    Args:
        destination (str, optional): Path to save the models. If None, uses the
            release's checkpoint directory (get_checkpoint_dir()).
        instruments (list, optional): Instruments whose TCN and rescore models
            to download ('qtof', 'orbitrap', 'timstof'). If None, all of them.
    """
    destination = os.path.abspath(destination or get_checkpoint_dir())
    os.makedirs(destination, exist_ok=True)
    print(f"Models will be downloaded to: {destination}")

    for instrument in instruments or INSTRUMENTS:
        if instrument not in INSTRUMENTS:
            print(f"Warning: Unknown instrument '{instrument}', skipping.")
            continue
        with tempfile.TemporaryDirectory() as temp_dir:
            for url in release_asset_urls(instrument):
                zip_path = os.path.join(temp_dir, url.rsplit("/", 1)[-1])
                print(f"Downloading {url.rsplit('/', 1)[-1]}...")
                try:
                    download_url(url, zip_path)
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                    print(f"You can manually download and extract it from {url}")
            for model_file in model_files(instrument):
                src_path = os.path.join(temp_dir, model_file)
                if os.path.exists(src_path):
                    dest_path = os.path.join(destination, model_file)
                    shutil.move(src_path, dest_path)
                    print(f"Installed {model_file} to {dest_path}")
                else:
                    print(f"Warning: {model_file} not found in the download")

    print("\nDownload complete!")
    print(f"Models are installed in: {destination}")


def check_models_exist():
    """Check if all models exist in the checkpoint directory."""
    model_dir = get_checkpoint_dir()
    return all(
        os.path.exists(os.path.join(model_dir, name))
        for instrument in INSTRUMENTS
        for name in model_files(instrument)
    )


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
        for instrument in INSTRUMENTS:
            for name in model_files(instrument):
                if os.path.exists(os.path.join(checkpoint_dir, name)):
                    print(f"  [+] {name}")
                else:
                    print(f"  [-] {name} (not downloaded)")
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
        "-i",
        "--instruments",
        nargs="+",
        choices=INSTRUMENTS,
        help="Instruments whose models to download (default: all)",
    )
    args = parser.parse_args()
    download_models(destination=args.destination, instruments=args.instruments)


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
