# Agreement with the reference implementation

These examples address reviewer R1's request (2b1) to *demonstrate exact
agreement with the reference in numerically stable examples* and to *report the
magnitude and source of any disagreement*.

## `envz_ompr_reference.jl` — exact agreement (numerically stable)

The EnvZ-OmpR two-component system is the canonical ACR example with an
analytically known answer that the MATLAB reference reproduces. COCOA returns
**exactly** the reference result:

| quantity | reference (paper S.5.2) | COCOA |
|---|---|---|
| ACR  | `{Yp}` | `{Yp}` |
| ACRR | 15 pairs over `{X, XD, XDYp, XT, XTYp, XpY}` | 15 pairs, same set |

```bash
julia --project examples/reference_comparison/envz_ompr_reference.jl
```

Use the **exhaustive** detector (`kinetic_efficient=false`): the fast default is
a lower bound and can miss this ACR (reviewer 2b3).

## Genome-scale comparison (in the revision workspace)

For genome-scale models the reference ships ACR/ACRR ground truth
(`MetSingle`/`MetDouble` CSVs). The comparison script lives with the revision
analysis (it needs the reference repo), not in the package:
`COCOA_revision/scripts_linux/compare_to_reference.jl`.

Headline (iJR904, "fixed" binding order):

- **ACR: reference = 0, COCOA = 0 for every seed and both detector modes →
  exact agreement.** (COCOA's ACR = 0 on iJR904 is *correct*, confirmed by the
  reference, not a detection failure.)
- **ACRR: reference = 10, COCOA = 131–1469** — a genuine magnitude difference
  attributable to the ACRR-detection strategy and elementary-splitting binding
  order (see `COCOA_revision/REVISION_ANALYSIS.md` §3a), not to a coding error.
