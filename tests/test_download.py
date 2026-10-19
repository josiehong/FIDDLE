import os

from msfiddle import download
from msfiddle.download import checkpoint_release_for_package_version


def test_checkpoint_release_for_2_0_1_reuses_2_0_0_assets():
    assert checkpoint_release_for_package_version("2.0.1") == "v2.0.0"


def test_checkpoint_release_2_0_and_2_1_share_2_0_0_assets():
    assert checkpoint_release_for_package_version("2.0.0") == "v2.0.0"
    assert checkpoint_release_for_package_version("2.1.0") == "v2.0.0"


def test_checkpoint_release_per_minor_version_from_2_2():
    assert checkpoint_release_for_package_version("2.2.0") == "v2.2.0"
    assert checkpoint_release_for_package_version("2.2.3") == "v2.2.0"
    assert checkpoint_release_for_package_version("2.3.0") == "v2.3.0"
    assert checkpoint_release_for_package_version("3.4.5") == "v3.4.0"


def test_checkpoint_release_handles_prerelease_versions():
    assert checkpoint_release_for_package_version("2.0.1.dev0") == "v2.0.0"
    assert checkpoint_release_for_package_version("2.2.0.dev0") == "v2.2.0"


def test_checkpoint_dir_is_keyed_by_release():
    assert os.path.basename(download.get_checkpoint_dir()) == download.FIDDLE_RELEASE
