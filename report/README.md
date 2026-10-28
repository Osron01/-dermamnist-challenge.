# Report

`report.tex` is the course's IEEE template (`IEEEtran.cls`, `IEEEtran.bst`, `bare_conf.tex` unchanged),
filled with our results. Every number in it comes from `results/summary.csv` and `results/test/`.

Compile: upload this folder to Overleaf (New Project -> Upload Project), or locally
`pdflatex report && bibtex report && pdflatex report && pdflatex report`.

Before handing in:
- add the TensorBoard screenshot as `figures/tensorboard.png` (`python scripts/to_tensorboard.py`,
  `tensorboard --logdir runs`); until then the figure shows a TODO box
- write the AI-use and contribution statements (marked TODO)
- check every entry of `references.bib` against the original paper
- check it stays within 4 pages (references do not count)
