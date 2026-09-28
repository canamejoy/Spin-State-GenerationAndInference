# Spec: Reorganize Research Repository

## Purpose

This change moves files and repairs references; it adds no new capability and modifies no
existing behavior (proposal: New Capabilities = None, Modified Capabilities = None). This spec
is therefore acceptance criteria, not feature requirements — each one is directly executable
and yields an unambiguous pass/fail.

## Requirements

### Requirement: No Stale Path References Survive

The repository MUST NOT contain any live reference to a pre-move path after the change.

#### Scenario: Repo-wide grep for stale tokens

- GIVEN the move map in `proposal.md` has been applied
- WHEN a repo-wide grep is run for `integration_cycle/`, `notebooks/replaced/`,
  `notebooks/cycle/ciclo_completo.md`, `notebooks/cycle/ciclo_external.md`
- THEN it returns zero hits in `README.md`, `docs/`, `notebooks/`, `tests/`, `.gitignore`,
  and `openspec/`
- AND `AGENT_PROMPT.md` and `AGENT_PROMPT_CORRECTIONS.md` are excluded from this check, since
  they are historical logs

### Requirement: Move History Is Preserved

Every path change MUST be a `git mv`, not a delete-and-recreate, so history survives.

#### Scenario: File exists once, at the new path

- GIVEN a file listed in the proposal's move map
- WHEN `git ls-files` is checked
- THEN the file exists at its new path and does not exist at its old path
- AND `git log --follow <new-path>` reaches commits authored before the move

### Requirement: Notebook Content Integrity

Moving a notebook MUST NOT alter its cell source.

#### Scenario: Content hash unchanged across the move

- GIVEN a `.ipynb` file moved by this change
- WHEN its content hash before the move is compared to its content hash after the move
- THEN the hashes are identical

### Requirement: Test Suite Passes and Is Tracked

`tests/` MUST be tracked by git and MUST pass after the reorganization.

#### Scenario: Test run is green

- GIVEN the reorganization is applied
- WHEN `.venv/bin/python -m pytest tests/ -q` is run
- THEN it reports 14 passed
- AND `git ls-files tests/` lists the test suite

### Requirement: Notebook Check Script Finds Notebooks

`_nbcheck.py`'s path template MUST resolve against the new layout, and its result MUST be
checked by count, not by exit code alone.

#### Scenario: Non-zero notebook count

- GIVEN `_nbcheck.py` has been updated for the new `notebooks/integration/` layout
- WHEN it is run
- THEN it exits zero
- AND it reports a notebook count greater than zero (a zero exit with a zero count is a FAIL)

### Requirement: Gitignore Carve-Out Correctness

The `.gitignore` rewrite MUST version the article source and result figures while keeping every
LaTeX byproduct ignored.

#### Scenario: Source and figures are trackable

- GIVEN the `.gitignore` carve-out is applied
- WHEN `git status --porcelain` and `git check-ignore` are run against `main/`
- THEN `main.tex`, `refs.bib`, `Makefile`, `PREDECESSOR_OUTLINE.md`, `README.md`,
  `figures/architectures.tex`, `report/results.tex`, `report/topology.tex`, and the four result
  PNGs are tracked or trackable
- AND every `.aux`, `.log`, `.out`, `.fls`, `.fdb_latexmk`, `.bbl`, `.blg`, `.spl`, and built
  `.pdf` under `main/` remains ignored

### Requirement: Documentation Matches Reality

Documentation MUST describe the notebooks that actually exist and their actual relationships.

#### Scenario: Integration README covers all eight notebooks

- GIVEN `notebooks/integration/README.md` after the rewrite
- WHEN its contents are reviewed
- THEN it documents all three training notebooks and all five evaluation notebooks

#### Scenario: Cycle README states its relationship to integration evaluations

- GIVEN `notebooks/cycle/README.md`
- WHEN its contents are reviewed
- THEN it exists and states the relationship between its three notebooks and the
  `notebooks/integration/evaluation/` notebooks

## Non-Goals (explicit)

- `notebooks/utils/metrics.py` MUST NOT move.
- `notebooks/generative/` MUST NOT be re-tiered by this change.
- No notebook MUST be deleted or archived beyond the confirmed `notebooks/replaced/` ->
  `notebooks/archive/` move already in the move map.
- The untracked foreign `01_baseline_simulation.ipynb` at the repo root MUST remain untouched.

## Open Question (carried, not resolved)

`main/report/toy.txt` was never inspected during exploration. Default rule: it stays ignored
unless explicitly classified as article source during apply. This is not a blocker for the
spec and MUST be revisited before `sdd-archive`.
