# Expected results

Reference values produced by `robustness_experiments.jl` on the current code, for checking a
local re-run. Seeds vary within each row. The giant kinetic module and ACR are invariant at
every setting, while the concordance partition itself varies from run to run without changing
them: it is a single partition on e_coli_core over the full cone and varies at genome scale (on
e_coli_core it also varies on the constrained biomass face and at a concordance tolerance of
10⁻¹⁰, below solver precision). ACRR reflects finite-sampling noise and converges with sample
size (iJR904: 1418 → 1378 pairs from 1000 to ≥ 5000 samples).

## e_coli_core

Defaults: full steady-state cone, concordance tolerance 10⁻², blocked-reaction tolerance
10⁻⁹, 1000 samples.

| Flux space | Conc. tol. | Blocked-rxn tol. | Transitivity | Runs | Concordance modules | Giant kinetic | ACR | ACRR |
|---|---|---|---|---:|---:|---:|---:|---:|
| full cone | 10⁻² | 10⁻⁹ | on, off | 20 | 109 | 4 | 0 | 1 |
| biomass face (0.999) | 10⁻² | 10⁻⁹ | on | 3 | 122 | 4 | 0 | 1 |
| full cone | 10⁻⁶ | 10⁻⁹ | on | 3 | 109 | 4 | 0 | 1 |
| full cone | 10⁻⁸ | 10⁻⁹ | on | 3 | 109 | 4 | 0 | 1 |
| full cone | 10⁻¹⁰ | 10⁻⁹ | on | 3 | 107–109 | 4 | 0 | 1 |
| full cone | 10⁻² | 10⁻⁶ | on | 3 | 109 | 4 | 0 | 1 |
| full cone | 10⁻² | 10⁻⁸ | on | 3 | 109 | 4 | 0 | 1 |
| full cone | 10⁻² | 10⁻¹⁰ | on | 3 | 109 | 4 | 0 | 1 |

## iJR904 (genome scale)

Full steady-state cone, concordance tolerance 10⁻².

| Sample size | CV threshold | Transitivity | Runs | Concordance modules | Giant kinetic | ACR | ACRR |
|---:|---|---|---:|---:|---:|---:|---:|
| 1000 | 0.01 | on, off | 17 | 664–676 | 106–107 | 0 | 1418–1460 |
| 5000 | 0.01 | on | 1 | 684 | 106 | 0 | 1378 |
| 20000 | 0.01 | on | 3 | 682–683 | 106 | 0 | 1378 |
| 1000 | 0.05 | on | 1 | 679 | 126 | 0 | 1896 |

## iAF1260b (genome scale)

Full steady-state cone, concordance tolerance 10⁻², seed × transitivity (one further run
discarded because its preprocessed network differed).

| Transitivity | Runs | Concordance modules | Giant kinetic | ACR | ACRR |
|---|---:|---:|---:|---:|---:|
| on | 10 | 1994–2002 | 40–78 | 0 | 770–1327 |
| off | 8 | 1998–2001 | 40 | 0 | 770–846 |

The giant kinetic module is 40 for every seed with transitivity off, and in all but one seed
with transitivity on (where a borderline sampled pair merges transitively to give 78).

ACR and ACRR denote the numbers of ACR metabolites and ACRR metabolite pairs; CV, coefficient
of variation. Small run-to-run differences in the partition (not in the giant module or ACR)
are expected at genome scale and reflect flux sampling, not a defect.
