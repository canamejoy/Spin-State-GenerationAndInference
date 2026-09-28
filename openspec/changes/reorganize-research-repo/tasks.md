# Tasks: Reorganize Research Repository

## Review Workload Forecast

| Field | Value |
|---|---|
| Estimated changed lines | Slice 1: 0 · Slice 2: ~210 · Slice 3: ~30 · Slice 4: ~1611+4 PNGs · Slice 5: 228 |
| 400-line budget risk | High (slice 4 only) |
| Chained PRs recommended | Yes (as chained commits — no PR practice in this repo) |
| Suggested split | 5 chained commits on `main`, applied in slice order, none pushed |
| Delivery strategy | auto-chain |
| Chain strategy | stacked-to-main (sequential commits on `main`; no branches, no PRs, no push) |

Decision needed before apply: No
Chained PRs recommended: Yes
Chain strategy: stacked-to-main
400-line budget risk: High

**`size:exception` granted for slice 4** (~1611 lines): scope-of-tracking import of
already-authored LaTeX article content that `.gitignore` was hiding, not newly written code.
Not sub-split further.

### Suggested Work Units

| Unit | Goal | Commit | Focused test command | Runtime harness | Rollback boundary |
|---|---|---|---|---|---|
| 1 | 13 `git mv`, zero edits | 1 | `git diff --cached -M --numstat -- '*.ipynb'` | N/A — pure rename, nothing executes | `git revert` this commit alone |
| 2 | Reference repair + `_nbcheck.py` fix | 2 | `rg` checks below + `_nbcheck.py` run | `.venv/bin/python notebooks/integration/_nbcheck.py` | `git revert` this commit alone |
| 3 | `.gitignore` rewrite | 3 | `git check-ignore -v` (trackable + byproduct sets) | N/A — no execution | `git revert` this commit alone |
| 4 | Track `main/` (13 files, `size:exception`) | 4 | `git ls-files main/ \| wc -l` | N/A — LaTeX not compiled here | `git revert` this commit alone |
| 5 | Track `tests/` | 5 | `.venv/bin/python -m pytest tests/ -q` | Same command is the runtime harness | `git revert` this commit alone |

**Global rules**: no notebook cell source is edited anywhere in this change. The only code
change in the entire change is `notebooks/integration/_nbcheck.py`. Never edit a file that is
itself in the proposal's move map before its `git mv`; edit it at its new path (applies to
`notebooks/archive/ciclo_external.md`, `notebooks/integration/README.md`,
`notebooks/integration/_nbcheck.py`). `docs/07_metrics.md` needs no edit — verified pointing at
`notebooks/utils/metrics.py`, which does not move.

## Phase 0: Baseline Capture (before any `git mv`)

- [x] 0.1 Run design.md's Testing Strategy check-2 hash script against
      `notebooks/integration/*.ipynb` and `integration_cycle/*.ipynb`, redirected to
      `/tmp/nbhash.before`, before touching any file.

## Phase 1: Move — Commit 1

- [x] 1.1 Execute all 13 `git mv` operations from `proposal.md`'s move map (no content edits).
- [x] 1.2 Gate: `git diff --cached -M --numstat -- '*.ipynb'` → every row `0␉0␉<old> => <new>`,
      12 rows.

## Phase 2: Reference Repair — Commit 2

- [x] 2.1 Edit `README.md` lines 58, 60-63, 66 (structure tree), 114, 120, 122, 134.
- [x] 2.2 Edit `docs/04_direct_problem.md` lines 152-153.
- [x] 2.3 Edit `docs/05_complete_cycle.md` lines 138, 144.
- [x] 2.4 Edit `notebooks/archive/ciclo_external.md` line 121 (post-move path; in move map).
- [x] 2.5 Rewrite `notebooks/integration/README.md` (post-move path; in move map) to cover all
      three training and all five evaluation notebooks.
- [x] 2.6 Create `notebooks/cycle/README.md` stating the relationship to
      `notebooks/integration/evaluation/`.
- [x] 2.7 RED (implicit): confirm `notebooks/integration/_nbcheck.py` (post-move path; old
      hardcoded `integration_cycle/{NB}.ipynb` template) fails to resolve the new layout — this
      is the natural post-move failure state, no synthetic test needed.
- [x] 2.8 GREEN: apply design.md Decision 5 exact head/tail to
      `notebooks/integration/_nbcheck.py`; `collect()` stays unchanged.
- [x] 2.9 Gate — content integrity: rerun the check-2 hash script to `/tmp/nbhash.after`, then
      `diff /tmp/nbhash.before /tmp/nbhash.after` → empty.
- [x] 2.10 Gate — stale refs (each scoped `rg -n ... --glob '!openspec/**' --glob
      '!AGENT_PROMPT*.md' --glob '!*.ipynb'`, no match/exit 1): `integration_cycle/`,
      `notebooks/replaced/`, `notebooks/cycle/ciclo_completo.md`,
      `notebooks/cycle/ciclo_external.md`, `04a_|04b_|04c_|04d_`.
- [x] 2.11 Gate — proof `_nbcheck.py` works: `.venv/bin/python notebooks/integration/_nbcheck.py`
      → prints `_nbcheck: 8 notebook(s)`, exit 0; `.venv/bin/python
      notebooks/integration/_nbcheck.py __nope__; echo $?` → `2`.

## Phase 3: `.gitignore` — Commit 3

- [x] 3.1 Replace `.gitignore` lines 53-60 with the exact block in design.md Decision 4 (delete
      the bare `main/` line; add extension-scoped byproduct rules; keep `papers/`; add
      `!main/report/*.png` / `!main/figures/*.png` after the blanket `*.png`).
- [x] 3.2 Gate — trackable: `git check-ignore -v <path>` exits 1 for each of the 13 `main/`
      source paths (article source, `README.md`, `Makefile`, `PREDECESSOR_OUTLINE.md`,
      `figures/architectures.tex`, `report/results.tex`, `report/topology.tex`, 4 result PNGs).
- [x] 3.3 Gate — still ignored: `git check-ignore -v <path>` exits 0 for `main/main.{pdf,log,
      aux,bbl,blg,fls,fdb_latexmk,spl,out}`, `main/report/{results,topology}.{pdf,aux,log,out}`,
      `main/report/b.log`, and `git check-ignore -v papers/`.

## Phase 4: Track `main/` — Commit 4 (`size:exception` granted)

- [x] 4.1 `git add main/` — 13 files including `main/report/toy.txt` (confirmed hand-authored
      source, not a byproduct — no special-case task, it is trackable by extension after 3.1).
- [x] 4.2 Gate: `git ls-files main/ | wc -l` → `13`.

## Phase 5: Track `tests/` — Commit 5

- [x] 5.1 `git add tests/conftest.py tests/test_topology.py`.
- [x] 5.2 Gate: `git ls-files tests/` → exactly those two paths.
- [x] 5.3 Gate: `.venv/bin/python -m pytest tests/ -q` → `14 passed`.

## Phase 6: Full-Range Verification (after all five commits)

- [x] 6.1 Gate: `git status --porcelain` → no tracked-file change left over; untracked set
      unchanged from baseline (`01_baseline_simulation.ipynb`, `.codegraph/`, `openspec/`).
- [x] 6.2 Gate: `git log --follow <new-path>` for one moved file reaches pre-move history.
