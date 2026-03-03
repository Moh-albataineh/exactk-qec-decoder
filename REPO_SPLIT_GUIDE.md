# Repository Split Guide

This guide explains how to publish `public_release_exactk_v1/` as a standalone GitHub repository.

---

## Option 1: New GitHub Repository (Recommended)

### Step 1: Create the repository on GitHub

Go to https://github.com/new and create `exactk-qec-decoder` (empty, no README).

### Step 2: Initialize and push

```bash
cd public_release_exactk_v1

# Initialize git
git init
git add .
git commit -m "ExactK V1.0: iso-K hinge loss for QEC decoding

- Factor-Graph v1 bipartite MP decoder
- ExactK iso-K hinge loss (ΔK=0, λ=0.10, margin=0.30)
- Selector v6 (drop_slice_floor) production policy
- MLOps: JSONL WAL + progressive checkpoints
- Validated: d=5 +31.2%, d=7 +23.4%, holdout +45.0%
- 80% safe yield, 0% TOPO_FAIL, 0 Do-No-Harm violations"

# Add remote and push
git remote add origin git@github.com:YOUR_USERNAME/exactk-qec-decoder.git
git branch -M main
git push -u origin main
```

### Step 3: Tag the release

```bash
git tag -a v1.0.0 -m "ExactK V1.0 Release - Holdout validated"
git push origin v1.0.0
```

### Step 4: Create GitHub Release

1. Go to `https://github.com/YOUR_USERNAME/exactk-qec-decoder/releases/new`
2. Select tag `v1.0.0`
3. Title: `ExactK V1.0 — Iso-K Hinge Loss for QEC Decoding`
4. Description: copy from `CHANGELOG.md` v1.0.0 entry
5. Check "Set as the latest release"
6. Publish

---

## Option 2: Subtree Split (Keep in Current Repo)

If you prefer to keep the release as a subdirectory in the existing `qec-noise-factory` repo:

### Step 1: Split the subtree

```bash
cd /path/to/qec-noise-factory
git subtree split --prefix=public_release_exactk_v1 -b exactk-v1-release
```

### Step 2: Push to a new remote

```bash
# Create new repo on GitHub first
git remote add exactk-release git@github.com:YOUR_USERNAME/exactk-qec-decoder.git
git push exactk-release exactk-v1-release:main
```

### Step 3: Tag on the new remote

```bash
git push exactk-release exactk-v1-release:refs/tags/v1.0.0
```

---

## GitHub Release Checklist

- [ ] Repository is public
- [ ] README renders correctly
- [ ] CITATION.cff is recognized by GitHub (shows "Cite this repository" button)
- [ ] LICENSE is recognized by GitHub (shows MIT badge)
- [ ] All links in README work (relative paths)
- [ ] `.gitignore` prevents artifact upload
- [ ] Tag `v1.0.0` exists
- [ ] GitHub Release page has changelog notes
- [ ] No `.pt`, `.pth`, `.jsonl` files in repo
- [ ] No personal paths (`C:\Users\...`) in code files
- [ ] Repository description set to: "ExactK: Iso-K Hinge Loss for K-Leakage-Free QEC Decoding"
- [ ] Topics set: `quantum-error-correction`, `surface-codes`, `machine-learning`, `gnn-decoder`
