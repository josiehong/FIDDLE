import os

import pytest

from msfiddle import download
from msfiddle.download import checkpoint_release_for_package_version


@pytest.mark.parametrize(
    "version, release",
    [
        ("2.0.0", "v2.0.0"),
        ("2.0.1.dev0", "v2.0.0"),
        ("2.1.0", "v2.0.0"),
        ("2.2.0", "v2.2.0"),
        ("2.2.3", "v2.2.0"),
        ("2.2.0.dev0", "v2.2.0"),
        ("2.3.0", "v2.3.0"),
        ("3.4.5", "v3.4.0"),
    ],
)
def test_checkpoint_release_for_package_version(version, release):
    assert checkpoint_release_for_package_version(version) == release


def test_checkpoint_dir_is_keyed_by_release():
    assert os.path.basename(download.get_checkpoint_dir()) == download.FIDDLE_RELEASE
