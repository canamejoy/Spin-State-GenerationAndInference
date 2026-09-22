# main/ — the article

LaTeX source for the direct-problem companion to
*Regression-based explainable deep learning for estimating Hamiltonian
parameters from magnetic nanodot images* (Materials & Design, 2026).

This folder is **gitignored**.

```
main.tex                 elsarticle skeleton, sectioned to mirror the predecessor
refs.bib                 bibliography (reading list: ../papers/README.md)
PREDECESSOR_OUTLINE.md   extracted structure of the predecessor paper — the template
figures/                 written by the notebooks via metrics.save_figure()
Makefile                 make / make watch / make clean
```

## Section hierarchy

`main.tex` mirrors the predecessor's structure one-for-one, so the two papers
read as a single programme:

| Predecessor | This article |
|---|---|
| 1. Introduction | 1. Introduction |
| 2. Related work | 2. Related work |
| 3.1 Atomistic Hamiltonian framework | 3.1 (same) |
| 3.2 GPU-accelerated image dataset generation | 3.2 (same) |
| 3.3 Deep regression framework | 3.3 Conditional diffusion framework |
| 3.4 Spatial interpretability via RAMs | 3.4 Cycle integration via the inverse latent |
| 4. Experimental set-up | 4. Experimental set-up |
| 5. Results and discussion (5.5 Limitations) | 5. Results and discussion (5.5 Limitations) |
| 6. Conclusions | 6. Conclusions |

`PREDECESSOR_OUTLINE.md` carries the full extraction: every section heading,
all 13 figure captions, all 4 table captions, the parameter table, the reported
per-parameter R²/MAE and the reference style. Consult it before writing any
section, and match its figure and table conventions.

## Build

```bash
make          # latexmk -pdf
make watch    # continuous preview
```

Requires a TeX distribution with `elsarticle`. On Debian/Ubuntu:
`sudo apt install texlive-publishers texlive-science latexmk`.

## Conventions inherited from the predecessor

- Notation macros are defined at the top of `main.tex` — use `\bth`, `\Kdm`,
  `\qpeak` etc. rather than writing the symbols out, so a notation change is one
  edit.
- Figures land in `figures/` from the notebooks. `metrics.save_figure()` writes
  PNG at 300 dpi and SVG; include the PDF/SVG in the article, never the PNG.
- `metrics.apply_figure_style()` sets the serif, 11 pt, top/right-spine-free
  style that matches the predecessor's figures. Call it before plotting.
- Every unwritten passage is marked `\todo{...}` and renders in red. Grep for
  `\todo` to see what is outstanding; comment out the macro's colour before
  submission.
