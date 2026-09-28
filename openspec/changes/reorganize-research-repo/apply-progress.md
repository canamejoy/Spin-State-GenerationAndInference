# Apply Progress: Reorganize Research Repository

**Change**: reorganize-research-repo
**Mode**: Strict TDD (single code change: `notebooks/integration/_nbcheck.py`, RED/GREEN in
same commit per tasks 2.7/2.8; every other slice is mechanical move/edit/import work with its
own machine-checkable gate, per Work Unit Evidence below)
**Status**: 24/24 tasks complete. All five chained commits landed on `main`, nothing pushed.

## Completed Tasks

- [x] 0.1 Baseline cell-source hash captured to `/tmp/nbhash.before` (8 hashes) before any `git mv`.
- [x] 1.1 All 13 `git mv` operations executed, zero content edits.
- [x] 1.2 Gate: `git diff --cached -M --numstat -- '*.ipynb'` — 12 rows, all `0␉0`.
- [x] 2.1 `README.md` structure tree + 4 stale-path lines updated.
- [x] 2.2 `docs/04_direct_problem.md` lines 152-153 updated.
- [x] 2.3 `docs/05_complete_cycle.md` lines 138, 144 updated.
- [x] 2.4 `notebooks/archive/ciclo_external.md` line 121 updated (post-move path).
- [x] 2.5 `notebooks/integration/README.md` rewritten — 3 training + 5 evaluation notebooks documented.
- [x] 2.6 `notebooks/cycle/README.md` created, states relationship to `notebooks/integration/evaluation/`.
- [x] 2.7 RED confirmed: old `_nbcheck.py` raised `FileNotFoundError` on `integration_cycle/...` after the move.
- [x] 2.8 GREEN: `_nbcheck.py` rewritten per design.md Decision 5 exact head/tail; `collect()` untouched.
- [x] 2.9 Gate: post-move hash multiset (`/tmp/nbhash.after`) diffed empty against baseline.
- [x] 2.10 Gate: 4/5 scoped greps zero hits; 5th token has one out-of-scope, non-stale hit — see Risks.
- [x] 2.11 Gate: `_nbcheck.py` prints `_nbcheck: 8 notebook(s) under ...`, exit 0; empty filter exits 2.
- [x] 3.1 `.gitignore` lines 53-60 replaced with design.md Decision 4's exact block.
- [x] 3.2 Gate: `git check-ignore` (canonical exit code) exits 1 for all 13 `main/` source paths.
- [x] 3.3 Gate: `git check-ignore -v` exits 0 for every listed byproduct and `papers/`.
- [x] 4.1 `git add main/` — 13 files including `main/report/toy.txt` (classified as source).
- [x] 4.2 Gate: `git ls-files main/ | wc -l` → `13`.
- [x] 5.1 `git add tests/conftest.py tests/test_topology.py`.
- [x] 5.2 Gate: `git ls-files tests/` → exactly those two paths.
- [x] 5.3 Gate: `pytest tests/ -q` → `14 passed`.
- [x] 6.1 Gate: `git status --porcelain` — only pre-existing unrelated modifications remain; untracked set is exactly `01_baseline_simulation.ipynb` + `openspec/`.
- [x] 6.2 Gate: `git log --follow` on a moved file reaches pre-move history (5 pre-move commits, back to `3776283 integration cycle`).

## Files Changed

| File | Action | Commit | What Was Done |
|---|---|---|---|
| `notebooks/replaced/*` (4 files) | `git mv` → `notebooks/archive/` | 1 | Rename only |
| `notebooks/cycle/ciclo_completo.md`, `ciclo_external.md` | `git mv` → `notebooks/archive/` | 1 | Rename only |
| `integration_cycle/*` (8 `.ipynb` + `README.md` + `_nbcheck.py`) | `git mv` → `notebooks/integration/{training,evaluation}/` | 1 | Rename only |
| `README.md` | Modified | 2 | Structure tree + 4 stale link lines |
| `docs/04_direct_problem.md` | Modified | 2 | 2 stale link lines |
| `docs/05_complete_cycle.md` | Modified | 2 | 2 stale reference lines |
| `notebooks/archive/ciclo_external.md` | Modified | 2 | 1 stale link line (post-move path) |
| `notebooks/integration/README.md` | Rewritten | 2 | Documents all 3 training + 5 evaluation notebooks |
| `notebooks/cycle/README.md` | Created | 2 | States relationship to `notebooks/integration/evaluation/` |
| `notebooks/integration/_nbcheck.py` | Modified | 2 | `__file__`-relative discovery, exit 2 on empty result |
| `.gitignore` | Modified | 3 | Deleted bare `main/` line; extension-scoped byproduct rules |
| `main/` (13 files) | Tracked (`git add`) | 4 | `size:exception` — LaTeX article import |
| `tests/conftest.py`, `tests/test_topology.py` | Tracked (`git add`) | 5 | Test suite import |

## Commits (chained, on `main`, none pushed)

1. `1d460d0` `refactor(notebooks): move integration_cycle and archived notebooks under notebooks/`
2. `090c45d` `docs(notebooks): repair references to the moved integration and archive paths`
3. `3b9e2aa` `fix(gitignore): delete bare main/ exclusion, ignore LaTeX byproducts by extension`
4. `0b951ea` `build(article): track main/ LaTeX source and result figures (size:exception)`
5. `5247d8c` `test: track tests/ so the pytest suite stops going unnoticed`

## TDD Cycle Evidence (Strict TDD — the one code change, `_nbcheck.py`)

| Task | RED | GREEN | REFACTOR |
|---|---|---|---|
| 2.7/2.8 `_nbcheck.py` layout fix | Ran old script post-move: `FileNotFoundError: integration_cycle/01_ddpm_cosine_frozen_encoder.ipynb` (exit 1, natural post-move failure, no synthetic test needed per design) | Applied design.md Decision 5 exact head/tail; ran against new layout: `_nbcheck: 8 notebook(s) under .../notebooks/integration`, exit 0; empty-filter run: `_nbcheck: FATAL: no notebooks found ... matching ['__nope__']`, exit 2 | `collect()` left byte-identical per design; no further refactor needed — landed in the same commit (2) as required |

All other tasks are mechanical move/edit/import work with no production logic — no RED/GREEN
applies; each has its own machine-checkable gate captured below and in Completed Tasks.

## Work Unit Evidence

| Unit | Focused test command and exact result | Runtime harness command/scenario and exact result | Rollback boundary |
|---|---|---|---|
| 1 (move) | `git diff --cached -M --numstat -- '*.ipynb'` → 12 rows, all `0␉0` | N/A — pure rename, nothing executes | `git revert 1d460d0` |
| 2 (refs + `_nbcheck.py`) | 5 scoped `rg` gates (4/5 zero-hit, 1 out-of-scope hit, see Risks) + hash-diff empty | `.venv/bin/python notebooks/integration/_nbcheck.py` → `_nbcheck: 8 notebook(s) under ...`, exit 0; `... __nope__; echo $?` → `2` | `git revert 090c45d` |
| 3 (`.gitignore`) | `git check-ignore` (canonical) exits 1 for 13 source paths, exits 0 for every byproduct + `papers/` | N/A — no execution | `git revert 3b9e2aa` |
| 4 (track `main/`) | `git ls-files main/ \| wc -l` → `13` | N/A — LaTeX not compiled here | `git revert 0b951ea` |
| 5 (track `tests/`) | `.venv/bin/python -m pytest tests/ -q` → `14 passed` | Same command is the runtime harness | `git revert 5247d8c` |

## Deviations from Design

None — implementation matches design.md exactly, including the byte-exact `.gitignore` block
(Decision 4) and the byte-exact `_nbcheck.py` head/tail (Decision 5).

## Issues Found

1. **Design's check-7 grep gate (`04a_|04b_|04c_|04d_`) has one hit outside the spec's checked
   scope.** `results/kaggle/README.md` (added in a recent, unrelated commit — `docs(results):
   rescue the raw Kaggle results into the repository`) documents real, on-disk Kaggle result
   artifact filenames (`04a_ddpm_base.json`, `04b_cos_frozen.json`, `04c_cos_joint.json`,
   `04d_latent_guidance.json` — confirmed present under `results/kaggle/`), not stale notebook
   paths. `results/` is not in spec.md's "No Stale Path References Survive" checked-directory
   list (`README.md`, `docs/`, `notebooks/`, `tests/`, `.gitignore`, `openspec/`), and
   design.md's File Changes table never lists `results/kaggle/README.md`, so it is out of this
   change's edit scope — renaming those filenames would break real artifact references. Left
   untouched; not a stale reference, but the design's own check-7 command is broader than the
   spec's requirement and will always report this one hit until `results/` is added to its
   glob excludes or the spec's scope is revisited.
2. **`git check-ignore -v` exit-code quirk for negated patterns.** With `-v`, git returns exit
   0 (not 1) for a path matched only by a negation rule (e.g. the 4 result PNGs under
   `main/report/*.png`), because `-v` counts any pattern match — negated or not — as exit 0;
   the printed line still shows the `!`-prefixed pattern, correctly indicating "not ignored."
   Verified empirically in an isolated throwaway repo and cross-checked against this repo's
   real paths. Gate 3.2 was additionally run with the canonical (non-`-v`) exit code, which
   correctly returns 1 for all 13 source paths including the 4 PNGs, and cross-checked against
   `git status --porcelain --ignored main/`, which shows `?? main/` (untracked/trackable) and
   lists only the true byproducts as `!!` (ignored). The functional requirement (spec:
   "Gitignore Carve-Out Correctness") is satisfied; this is a documentation note about the `-v`
   flag's exit-code semantics, not a gate failure.

## Remaining Tasks

None. 24/24 complete.

## Workload / PR Boundary

- Mode: chained commits (`stacked-to-main`), `size:exception` granted for commit 4 (~1611 lines).
- Current work unit: all 6 phases, complete.
- Boundary: commit 1 (`1d460d0`) through commit 5 (`5247d8c`), directly on `main`, nothing pushed.
- Estimated review budget impact: commits 1/3/4/5 are trivially reviewable (0, ~30, size-exception, 228 lines); commit 2 (~210 authored lines) is the only one needing normal review attention.

## Status

24/24 tasks complete. Ready for verify.
