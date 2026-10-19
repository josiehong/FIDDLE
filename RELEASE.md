# Release Checklist

Follow these steps every time you want to publish a new `msfiddle` version to PyPI. The package is built from `msfiddle/` in this repository.

## 1. Update the version

In `setup.py`, bump the version number:

```python
version="2.2.0",  # e.g. 2.1.0 → 2.2.0
```

`setup.py` is the single source of truth — `msfiddle.__version__` is derived
from the installed package metadata via `importlib.metadata`, so no other
file needs editing.

Follow [semantic versioning](https://semver.org/):
- `x.y.Z` — bug fixes
- `x.Y.0` — new features, backwards compatible
- `X.0.0` — breaking changes

Checkpoints follow the minor version (see step 6): every `x.y.z` release
downloads from the `vx.y.0` GitHub release, so a patch bump reuses the
existing checkpoints and a minor bump needs its own upload. 2.0.x and 2.1.x
are the exception; both use the `v2.0.0` assets.

## 2. Update CHANGELOG.md

Move the `[Unreleased]` items into a new versioned entry at the top of `CHANGELOG.md`:

```markdown
## [2.2.0] - YYYY-MM-DD
### Added
- ...
### Changed
- ...
### Fixed
- ...
```

## 3. Commit and push to main

```bash
git add setup.py CHANGELOG.md
git commit -m "Release v2.2.0"
git push origin main
```

## 4. Create a GitHub Release

Go to this repo on GitHub → **Releases** → **Draft a new release**:

1. Click **"Choose a tag"** → type `v2.2.0` → **"Create new tag: v2.2.0 on publish"**
2. Set title to `v2.2.0`
3. Paste the CHANGELOG entry into the description
4. Click **Publish release**

Any `v*` tag triggers the PyPI publish workflow, so only tag when you intend to release the package.

## 5. Verify

- Check the **Actions** tab in the repo to confirm the workflow succeeded
- Check [pypi.org/project/msfiddle](https://pypi.org/project/msfiddle/) to confirm the new version is live
- Test the new release locally:

```bash
pip install msfiddle==2.2.0
```

## 6. Checkpoints (every minor release)

`msfiddle` downloads checkpoints from the `vx.y.0` release matching its installed version (`checkpoint_release_for_package_version` in `msfiddle/download.py`) and caches them under `~/.msfiddle/check_point/vx.y.0/`. A minor release therefore needs four zips attached to its `vx.y.0` GitHub release, named exactly:

- `fiddle_tcn_qtof.zip`, `fiddle_rescore_qtof.zip`
- `fiddle_tcn_orbitrap.zip`, `fiddle_rescore_orbitrap.zip`

Build them from the training checkpoints with:

```bash
python scripts/package_release_checkpoints.py --tag 082426_muon --out_dir ./release_v2.2.0
```

`--tag` is the suffix of `check_point/fiddle_<model>_<tag>.pt`. The script keeps only the weights inference needs, writes one `fiddle_<model>.pt` per zip, and checks each zip against its source. If the model weights did not change, upload the previous release's zips again under the new tag.

Then check the download path from a clean install:

```bash
pip install msfiddle==2.2.0
msfiddle-download-models
msfiddle --demo --result_path ./output_demo.csv --no_cuda
```
