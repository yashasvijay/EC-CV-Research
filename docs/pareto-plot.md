# HOTA versus inference reduction figure

After the full search completes, run from the repository root:

```bash
.venv-eval/bin/python -m scripts.plot_pareto
```

The evaluation environment already provides Matplotlib. No plotting package
installation or runtime change is needed in the environment used by the active
search. The script uses a headless rendering backend.

Default inputs are `results/grid_search/summary.csv` and
`results/grid_search/pareto_frontier.json`. It also reads `summary.json` for
completion status and the evaluated full-inference baseline. All CSV/JSON
configuration IDs and plotted values are checked for consistency.

Default output is **`reports/ablations/pareto_frontier.png`**, at 300 DPI
(2400x1650 pixels). A sibling `pareto_frontier.plot.json` stores input checksums,
completion counts, baseline HOTA and plotted frontier coordinates.

## Interpreting the figure

- Horizontal axis: inference reduction (%), higher is better.
- Vertical axis: HOTA (%), higher is better.
- Gray points: all evaluated configurations in the CSV.
- Teal outlines: exported quality-frontier candidates, whose search objectives
  include latency in addition to HOTA and inference reduction.
- Orange line: the non-dominated frontier for the **two plotted objectives**.
  Both HOTA and reduction are maximized. Duplicate coordinates are collapsed
  only for line drawing; all configurations remain in the scatter layer.
- Star at x=0: the evaluated full-inference baseline. It is included in the
  frontier line only if it is non-dominated in this two-axis comparison.

The exported three-objective frontier may contain points dominated in the
HOTA/reduction projection because they offer lower latency. The line is
therefore calculated from all measured CSV points plus baseline instead of
blindly connecting the exported three-objective list. Connecting segments are
visual guides, not measurements of intermediate settings. With only one
non-dominated coordinate, the frontier is a point and has no connecting segment.

## Options

```bash
.venv-eval/bin/python -m scripts.plot_pareto \
  --input-dir results/grid_search \
  --output reports/ablations/pareto_frontier.png \
  --dpi 300 \
  --title 'MOT17-04-FRCNN: accuracy and inference reduction'
```

If the search did not include a baseline, supply a complete matching evaluated
baseline with `--baseline-summary results/baseline-04`. Its sequence, model,
frame count and full-inference fraction must match the search. Missing HOTA
values produce an error rather than a fabricated zero.

By default the search must have status `complete` and matching planned/completed
run counts. `--allow-partial` explicitly produces an incomplete preview with
`PARTIAL PREVIEW` and run counts in the title. Use a separate output filename
for previews to avoid mistaking them for the final publication figure.

No active grid-search files or processes are modified by plotting. Do not use
partial tuning results as final held-out accuracy evidence.
