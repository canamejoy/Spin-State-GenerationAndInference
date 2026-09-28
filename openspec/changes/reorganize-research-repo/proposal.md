# Proposal: Reorganize Research Repository

## Intent

Repository structure no longer reflects what sustains the paper. Verified problems:

- Three overlapping notebook layers: `notebooks/{inverse,generative,cycle,evaluation}`, `integration_cycle/`, and the frozen `notebooks/replaced/`.
- `integration_cycle/README.md` documents only `01`-`03`. It is silent on `04a`-`04d` and `05`, the notebooks that produced every number in `main/report/results.tex`.
- `notebooks/cycle/ciclo_completo.md` and `ciclo_external.md` document notebooks that were archived into `notebooks/replaced/`; `README.md` still links them as current.
- `main/` is gitignored wholesale, so the LaTeX article is unversioned.
- `tests/` (14 passing tests) is untracked — which is why the staleness above went unnoticed.

## Scope

### In Scope

- The move map below, executed as `git mv`. No notebook cell source is edited.
- Rewrite `notebooks/integration/README.md` to cover both the training and evaluation series.
- New `notebooks/cycle/README.md` stating its relationship to the integration evaluations.
- Update stale paths in `README.md` (lines 102, 108, 114, 120, 134), `docs/05_complete_cycle.md` (134-136, 144), `docs/07_metrics.md` (471).
- Fix the hardcoded `integration_cycle/{NB}.ipynb` template in `_nbcheck.py`.
- `.gitignore` rewrite: version the `main/` article source and result figures, keep every LaTeX byproduct ignored (exact diff in `explore.md`).
- Bring `tests/` into git.

### Out of Scope

- The untracked foreign `01_baseline_simulation.ipynb` at the repo root — the user relocates it to its own repository.
- Scientific and experimental-design work, planned separately.
- Archiving `notebooks/cycle/ciclo_completo_v2.ipynb` (confirmed: keep).
- Re-tiering `notebooks/generative/` (needs a provenance judgement).
- Chasing renames through `AGENT_PROMPT.md` / `AGENT_PROMPT_CORRECTIONS.md`; mark them historical instead.
- Updating Kaggle-hosted kernel copies pushed by slug.

## Capabilities

### New Capabilities

None — reorganization and documentation only.

### Modified Capabilities

None — no spec-level behavior changes.

## Approach

One mechanical pass: move first, then repair every reference, then verify by grep. Everything is text and path manipulation, so the change is fully reviewable without executing a notebook.

## Move map (authoritative)

Supersedes the `explore.md` Tier-1 table for all `integration_cycle/` rows.

| Source | Destination |
|---|---|
| `notebooks/replaced/` | `notebooks/archive/` |
| `notebooks/cycle/ciclo_completo.md` | `notebooks/archive/ciclo_completo.md` |
| `notebooks/cycle/ciclo_external.md` | `notebooks/archive/ciclo_external.md` |
| `integration_cycle/01_ddpm_cosine_frozen_encoder.ipynb` | `notebooks/integration/training/01_ddpm_cosine_frozen_encoder.ipynb` |
| `integration_cycle/02_ddpm_cosine_joint_finetune.ipynb` | `notebooks/integration/training/02_ddpm_cosine_joint_finetune.ipynb` |
| `integration_cycle/03_latent_guided_sampler.ipynb` | `notebooks/integration/training/03_latent_guided_sampler.ipynb` |
| `integration_cycle/04a_ddpm_base.ipynb` | `notebooks/integration/evaluation/01_ddpm_base.ipynb` |
| `integration_cycle/04b_cos_frozen.ipynb` | `notebooks/integration/evaluation/02_cos_frozen.ipynb` |
| `integration_cycle/04c_cos_joint.ipynb` | `notebooks/integration/evaluation/03_cos_joint.ipynb` |
| `integration_cycle/04d_latent_guidance.ipynb` | `notebooks/integration/evaluation/04_latent_guidance.ipynb` |
| `integration_cycle/05_cross_encoder_test.ipynb` | `notebooks/integration/evaluation/05_cross_encoder_test.ipynb` |
| `integration_cycle/README.md` | `notebooks/integration/README.md` (rewritten) |
| `integration_cycle/_nbcheck.py` | `notebooks/integration/_nbcheck.py` (template fixed) |

## What does not change

| Item | Why |
|---|---|
| `notebooks/utils/metrics.py` | Kaggle notebooks resolve it at runtime via `find_file()` against the Kaggle dataset `carloscanamejoy/physicalmetrics`, never from git. Moving it would mean editing `conftest.py` plus a hardcoded string in 8 notebooks for no benefit. |
| `notebooks/cycle/`, `notebooks/evaluation/` | Keep their notebooks; `ciclo_completo_v2.ipynb` is not a clean duplicate (`K_ENS = 1`). Documented, not moved. |
| `notebooks/generative/` | Not re-tiered in this change. |

## Verification

- **Primary**: repo-wide grep for every stale path — `integration_cycle/`, `notebooks/replaced/`, `ciclo_completo.md`, `ciclo_external.md` — must return zero live references.
- `.venv/bin/python -m pytest tests/ -q` — 14 tests, expected green.
- `_nbcheck.py` static check (`ast`/`json` only) against the new layout.
- No GPU and no Kaggle session are required, because no notebook cell source is touched.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Folding `integration_cycle/` under `notebooks/` enlarges the blast radius; a cross-reference is missed | Med | Grep is the acceptance gate, run for each stale token separately |
| `_nbcheck.py` path template silently matches nothing after the move | Med | Run it and assert a non-zero notebook count |
| Kaggle-hosted kernels pushed by slug still carry old paths | High | Operational follow-up, out of scope; recorded in the new README |
| `.gitignore` carve-out accidentally commits LaTeX byproducts | Low | `git status --porcelain` after the rewrite must list only source and the four result PNGs |
| `main/report/toy.txt` was never inspected | Low | Classify during apply; default to ignored |

## Rollback

Every step is a `git mv` or a text edit in one commit series on a branch. `git revert` the range, or `git checkout main -- .` before merge. No data, checkpoint, or Kaggle state is mutated, so rollback is complete and lossless.

## Dependencies

None. Repo-local `.venv` already exists and the test runner is verified working.

## Success Criteria

- [ ] Grep for `integration_cycle/`, `notebooks/replaced/`, `ciclo_completo.md`, `ciclo_external.md` returns no live references.
- [ ] `notebooks/integration/README.md` documents all three training and all five evaluation notebooks.
- [ ] `notebooks/cycle/README.md` exists and explains the relationship to the integration evaluations.
- [ ] `git ls-files main/` lists the article source and result figures; no `.aux`, `.log`, `.pdf` or other byproduct is tracked.
- [ ] `git ls-files tests/` lists the test suite.
- [ ] `.venv/bin/python -m pytest tests/ -q` passes 14/14.
- [ ] `_nbcheck.py` runs against the new layout and reports a non-zero notebook count.
- [ ] No `.ipynb` appears in the diff with changed cell source.
