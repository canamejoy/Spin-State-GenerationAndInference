# Exploration: reorganize-research-repo

Phase: `sdd-explore` · Status: complete · Store: hybrid (Engram topic `sdd/reorganize-research-repo/explore`, observation id 71)

## Scope

Reorganize this research repository so its structure reflects what actually sustains the
paper. This is a rename/move/document change. It touches no notebook cell source, no model
architecture, no loss function, no metric definition, and no scientific content.

## Current state

Three notebook layers overlap in purpose:

- `notebooks/{inverse,generative,cycle,evaluation,replaced,utils}` — the model-development tree.
- `integration_cycle/{01-03,04a-04d,05}` — the paper's real evidence layer.
- `notebooks/replaced/` — an already-frozen archive (4 notebooks, marked "do not edit").

`integration_cycle/README.md` documents only `01`-`03`. Notebooks `04a`-`04d` and `05`
produced every number in `main/report/results.tex` and are undocumented there. The flat
numeric sequence also conflates two roles: `01`-`03` are training runs, `04a`-`04d` and `05`
are evaluations.

**Second staleness bug (verified).** `notebooks/cycle/ciclo_completo.md` documents
`ciclo_completo_resultadosxclusters.ipynb` and `notebooks/cycle/ciclo_external.md` documents
`ciclo_external_dataset.ipynb`. Both notebooks live in `notebooks/replaced/`. The docs never
followed the archival move, and `README.md` lines 114 and 120 link to them as if they
described files still in `notebooks/cycle/`.

## metrics.py and its external consumers

`notebooks/utils/metrics.py` is the repo-wide single source of truth, but Kaggle and Colab
notebooks never import it from git at runtime — they resolve it via `find_file("metrics.py")`
against the separately maintained Kaggle dataset `carloscanamejoy/physicalmetrics` (verified:
38 references to that slug, 8 notebooks carrying a `_METRICS_LOCAL` fallback).

An in-repo move therefore has zero effect on any Kaggle run. The only coupling is
local-dev-facing: `tests/conftest.py` inserts `notebooks/utils` on `sys.path`, and each
`_METRICS_LOCAL` string hardcodes the same path.

**Decision: leave `notebooks/utils/metrics.py` exactly where it is.** Moving it would require
editing `conftest.py` plus a string inside 8 notebooks, for no gain toward this change's goal.

## Proposed rename/move map (Tier 1 — mechanical)

| Source | Destination | Rationale |
|---|---|---|
| `notebooks/replaced/` | `notebooks/archive/` | "replaced" is ambiguous; pure `git mv`, history preserved. |
| `notebooks/cycle/ciclo_completo.md` | `notebooks/archive/` | Follows the notebook it documents. |
| `notebooks/cycle/ciclo_external.md` | `notebooks/archive/` | Follows the notebook it documents. |
| `integration_cycle/01_ddpm_cosine_frozen_encoder.ipynb` | `integration_cycle/training/01_ddpm_cosine_frozen_encoder.ipynb` | Separates the training role from the evaluation role by folder rather than by a shared flat numbering. |
| `integration_cycle/02_ddpm_cosine_joint_finetune.ipynb` | `integration_cycle/training/02_ddpm_cosine_joint_finetune.ipynb` | Same. |
| `integration_cycle/03_latent_guided_sampler.ipynb` | `integration_cycle/training/03_latent_guided_sampler.ipynb` | Same. |
| `integration_cycle/04a_ddpm_base.ipynb` | `integration_cycle/evaluation/01_ddpm_base.ipynb` | Independently numbered evaluation series. |
| `integration_cycle/04b_cos_frozen.ipynb` | `integration_cycle/evaluation/02_cos_frozen.ipynb` | Evaluates `training/01`. |
| `integration_cycle/04c_cos_joint.ipynb` | `integration_cycle/evaluation/03_cos_joint.ipynb` | Evaluates `training/02`. |
| `integration_cycle/04d_latent_guidance.ipynb` | `integration_cycle/evaluation/04_latent_guidance.ipynb` | Evaluates `training/03`. |
| `integration_cycle/05_cross_encoder_test.ipynb` | `integration_cycle/evaluation/05_cross_encoder_test.ipynb` | Already an evaluation; only the folder changes. |
| `01_baseline_simulation.ipynb` (untracked, repo root) | removed or relocated by the user | Foreign: imports `src/damcmc`, targets the `Diffusion-Accelerated-MCMC` project. Needs explicit user confirmation. |

## Blast radius

Every reference that a move invalidates:

- `README.md` lines 102, 108, 114, 120, 134
- `docs/05_complete_cycle.md` lines 134-136, 144
- `docs/07_metrics.md` line 471
- `integration_cycle/README.md` — full rewrite required (silent on `04a`-`04d` and `05`)
- `integration_cycle/_nbcheck.py` — hardcodes `integration_cycle/{NB}.ipynb`; breaks when
  notebooks move into subfolders
- `tests/conftest.py` — hardcodes `notebooks/utils` (unaffected by this map, since
  `metrics.py` does not move)
- `AGENT_PROMPT.md`, `AGENT_PROMPT_CORRECTIONS.md` — historical build logs. Recommend marking
  them historical rather than chasing renames through them.
- `results/kaggle/README.md` — narrative mention of `metrics.py` only
- `.gitignore`

## .gitignore change

The current file ignores `main/` wholesale and also carries blanket `*.png` and `*.pdf` rules.
Un-ignoring the article requires a carve-out so the result figures survive while every LaTeX
byproduct stays ignored:

```gitignore
# LaTeX build artifacts (the article source itself is versioned)
main/*.aux
main/*.bbl
main/*.blg
main/*.fdb_latexmk
main/*.fls
main/*.log
main/*.out
main/*.pdf
main/*.spl
main/report/*.aux
main/report/*.log
main/report/*.out
main/report/*.pdf

# Result figures embedded in the article ARE versioned
!main/report/*.png
!main/figures/*.png
```

This versions `main.tex`, `refs.bib`, `Makefile`, `PREDECESSOR_OUTLINE.md`, `README.md`,
`figures/architectures.tex`, `report/results.tex`, `report/topology.tex` and the four result
PNGs. `main/report/toy.txt` was not inspected — unresolved.

## What is NOT moved, and why

`notebooks/cycle/` and `notebooks/evaluation/` stay in place. The "largely superseded" framing
is only partly true:

- `ciclo_completo_v2.ipynb` carries `K = 32` and `FAST_SAMPLING_STEPS = 100`, matching the
  shared protocol, **but also a `K_ENS = 1`** — so it is not a clean duplicate of
  `integration_cycle/evaluation/01_ddpm_base.ipynb`. Archival candidate, not a decided one.
- `cycle_complete_newmetrics.ipynb` uses a different, cheaper protocol (`K_ENS=16`,
  `TEST_FRACTION=0.10`).
- `ciclo_texture_fidelity.ipynb` performs a Hex-regime texture-fidelity analysis replicated
  nowhere in `integration_cycle/`.

Recommend adding a short README to `notebooks/cycle/` stating this relationship. Archival or
deletion of tracked history is a product decision and is not assumed here.

`notebooks/generative/` is not split into production vs experiment tiers in this change:
establishing which CVAE-Xception run produced the checkpoint consumed by
`generative_comparison.ipynb` requires a provenance judgement outside this change's scope.

## Sequencing and risk

- **Tier 1 — safe, mechanical, no execution needed.** Every `git mv`, all doc and
  `.gitignore` text edits, the foreign-notebook removal. No `.ipynb` cell source is touched,
  so verification is a repo-wide grep for stale paths.
- **Tier 2 — needs a smoke check, no GPU.** `integration_cycle/_nbcheck.py`'s hardcoded path
  template. Pure `ast`/`json` static analysis; runnable against the repo-local `.venv`.
- **Tier 3 — requires Kaggle.** None, provided no notebook cell source is edited.

Kaggle-hosted kernel copies pushed by slug are not updated by any `git mv` here. That is an
operational follow-up, not a blocker for this change.

## Product decisions — CONFIRMED by the user 2026-09-21

1. **Fold `integration_cycle/` under `notebooks/`: YES.** The user overrode the exploration's
   recommendation. One notebook tree. This enlarges the blast radius — every cross-reference
   to `integration_cycle/` moves in the same change — and the rename map below supersedes the
   Tier-1 table above for these paths:
   - `integration_cycle/01..03` -> `notebooks/integration/training/01..03`
   - `integration_cycle/04a..04d` -> `notebooks/integration/evaluation/01..04`
   - `integration_cycle/05_cross_encoder_test` -> `notebooks/integration/evaluation/05_cross_encoder_test`
   - `integration_cycle/README.md` -> `notebooks/integration/README.md` (full rewrite required)
   - `integration_cycle/_nbcheck.py` -> `notebooks/integration/_nbcheck.py` (hardcoded path
     template must be updated)
2. **Archive `notebooks/cycle/ciclo_completo_v2.ipynb`: NO.** Keep it in place. Verified it is
   not a clean duplicate — it carries `K = 32` and `FAST_SAMPLING_STEPS = 100` matching the
   shared protocol, but also a `K_ENS = 1`. Add a short `notebooks/cycle/README.md` stating
   the relationship to the integration evaluations.
3. **The untracked foreign `01_baseline_simulation.ipynb`: DO NOT TOUCH.** The user will
   relocate it to its own repository. It stays at the repo root for now and is out of scope
   for this change.

## Verification notes

- Confirmed by the orchestrator: the two misplaced `.md` files, the `find_file`/Kaggle-dataset
  resolution path, and the partial (not exact) protocol match of `ciclo_completo_v2.ipynb`.
- `ncsn_v2_adagn.ipynb` and `ddpm-spines-newlossfunction.ipynb` could not be opened (minified
  single-line JSON exceeds the read tool's ceiling). Their classification rests on filename
  and cross-reference inference only.
- `physical_metrics_comparison.ipynb` and `physical-metrics-comparison-4models.ipynb` were not
  opened; their near-duplicate names are flagged, not resolved.
