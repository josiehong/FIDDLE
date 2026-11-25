# Release Checklist

Follow these steps every time you want to publish a new `msfiddle` version to PyPI. The package is built from `msfiddle/` in this repository.

## 1. Update the version

In `setup.py`, bump the version number:

```python
version="2.3.0",  # e.g. 2.2.1 → 2.3.0
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
## [2.3.0] - YYYY-MM-DD
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
git commit -m "Release v2.3.0"
git push origin main
```

## 4. Create a GitHub Release

Go to this repo on GitHub → **Releases** → **Draft a new release**:

1. Click **"Choose a tag"** → type `v2.3.0` → **"Create new tag: v2.3.0 on publish"**
2. Set title to `v2.3.0`
3. Paste the CHANGELOG entry into the description
4. Click **Publish release**

Any `v*` tag triggers the PyPI publish workflow, so only tag when you intend to release the package. For a minor release, attach the checkpoint zips (step 6) to the release before publishing it: the new version is on PyPI a minute later, and `msfiddle-download-models` fails until the zips are there. From the command line, `gh release create` attaches them in the same step:

```bash
gh release create v2.5.0 release_v2.5.0/fiddle_{qtof,orbitrap,timstof}.zip \
    --target main --title "v2.5.0" --notes-file release_v2.5.0/RELEASE_NOTES.md
```

## 5. Verify

- Check the **Actions** tab in the repo to confirm the workflow succeeded
- Check [pypi.org/project/msfiddle](https://pypi.org/project/msfiddle/) to confirm the new version is live
- Check that Read the Docs built `latest` and `stable` from the release commit (project → **Builds**), and that [the stable docs](https://msfiddle.readthedocs.io/en/stable/) show the new version. Pushes trigger the builds through a webhook, set up once in the Read the Docs project under **Settings → Integrations** (GitHub incoming webhook); GitHub lists it under the repository's **Settings → Webhooks**, with its recent deliveries. Without it, build both versions by hand with **Build version**.
- Test the new release locally:

```bash
pip install msfiddle==2.3.0
```

## 6. Checkpoints (every minor release)

`msfiddle` downloads checkpoints from the `vx.y.0` release matching its installed version (`checkpoint_release_for_package_version` in `msfiddle/download.py`) and caches them under `~/.msfiddle/check_point/vx.y.0/`. A minor release therefore needs one zip per instrument attached to its `vx.y.0` GitHub release, named exactly:

- `fiddle_qtof.zip`, `fiddle_orbitrap.zip`, `fiddle_timstof.zip`

Each holds the instrument's TCN and rescore model (`fiddle_tcn_<instrument>.pt` and `fiddle_rescore_<instrument>.pt`), so the pair that belongs together is always downloaded together. Releases before 2.3 used one zip per model.

Build them from the training checkpoints with:

```bash
python scripts/package_release_checkpoints.py --tag v2.5_muon --out_dir ./release_v2.5.0
```

`--tag` is the suffix of `check_point/fiddle_<model>_<instrument>_<tag>.pt`; `--instruments` packages some of them, for example on the server that trained them. The script keeps only the weights inference needs, checks that each rescore model was trained with the TCN it is packaged with (by a hash of the TCN's weights), and checks each zip against its sources. If the model weights did not change, upload the previous release's zips again under the new tag.

Then check the download path from a clean install:

```bash
pip install msfiddle==2.3.0
msfiddle-download-models
msfiddle --demo --result_path ./output_demo.csv --no_cuda
```

## 7. Benchmarks (every minor release)

With the new checkpoints unpacked as `check_point/fiddle_{tcn,rescore}_<instrument>_v<x.y.0>.pt`, and BUDDY and SIRIUS installed (see `docs/benchmarks.rst`):

```bash
MODEL_TAG=v2.5.0 bash running_scripts/experiments_test_benchmark.sh
MODEL_TAG=v2.5.0 DATA_TAG=v2.5 MANIFEST=./data/benchmark/unseen_v2.5.0.csv \
    bash running_scripts/experiments_test_benchmark_unseen.sh --rebuild_manifest
cp result/benchmarks.png docs/_static/benchmarks.png
```

Run the unseen script where the training data are (`data/cl_pkl_<data tag>/`): `--rebuild_manifest` splits by them and writes the release's manifest of unseen compounds. Commit it (`data/benchmark/unseen_v<x.y.0>.csv`) with the figure. Without the flag, the script splits by an existing manifest. The scripts write only to `result/` and `data/`, so running them never changes the documentation.

Also export the timsTOF compound split there and attach it to the GitHub release (about 5 MB, too large for git). Only the timsTOF split is public; the Q-TOF and Orbitrap splits include NIST data:

```bash
python scripts/export_split_manifest.py --pkl_dir ./data/cl_pkl_v2.5 \
    --out ./timstof_split_v2.5.0.csv.gz
gh release upload v2.5.0 timstof_split_v2.5.0.csv.gz
```

