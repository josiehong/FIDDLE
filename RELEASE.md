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

Checkpoints are pinned to the major version (see step 6): all `x.y.z`
releases share the FIDDLE `x.0.0` checkpoint assets, so minor and patch
bumps do **not** require new checkpoint uploads.

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

## 6. Checkpoints (if model weights changed)

If new checkpoint files (`.pt`) were retrained, attach the zipped checkpoints to the `x.0.0` GitHub release for the current major version. `msfiddle` derives the checkpoint download URL from the installed package major version — for example, all `msfiddle` 2.x.x releases download checkpoint assets from this repo's `v2.0.0` release.
