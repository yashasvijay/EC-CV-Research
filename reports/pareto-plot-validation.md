# Pareto figure validation

Date: 2026-10-12

Four tests passed for dominance/tied coordinates and baseline inclusion,
partial-result refusal, missing baseline/inconsistent-export rejection, and
actual headless PNG rendering through the existing evaluation environment.

The PNG was checked for its signature and dimensions (2400x1650 at 300 DPI),
and the provenance JSON was checked against independently known synthetic
frontier coordinates. Visual inspection found a title-layout clipping issue;
the layout rectangle was corrected and the render test passed again. The final
synthetic figure was visually inspected for title, labels, legend, baseline,
frontier markers and explanatory caption.

The synthetic figure is a correctness/layout fixture, not a real grid-search
result. The real search was still active at implementation time. The final
`reports/ablations/pareto_frontier.png` was not generated from incomplete data.
No active source, search records, processes or runtime packages were modified.
