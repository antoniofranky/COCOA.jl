# Elementary Reaction Naming Convention

## Overview

This document describes the systematic naming convention for elementary reactions and enzyme-substrate complexes in COCOA.jl, designed following best practices from [Pham et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6409771/).

## Guiding Principles

1. **Unambiguous**: Each identifier uniquely maps to one entity
2. **Systematic**: Follow consistent, rule-based patterns
3. **Parseable**: Machine-readable structure for automated processing
4. **Human-readable**: Contain semantic information about the entity
5. **Database-compatible**: Compatible with BiGG Models and MetaNetX conventions
6. **Compartment-aware**: Include compartment information to avoid ambiguity

## Enzyme-Substrate Complex Identifiers

### Format
```
CPLX_E<enzyme_number>__<metabolite1>__<metabolite2>__..._<compartment>
```

### Components
- **`CPLX`**: Systematic prefix indicating "Complex" (enzyme-substrate intermediate)
- **`E<enzyme_number>`**: Enzyme identifier (e.g., `E3`, `E42`)
- **`__`** (double underscore): Separator between logical groups
- **`<metabolite_ids>`**: Sorted list of metabolite IDs joined by `__`
- **`_<compartment>`**: Single-letter compartment suffix (e.g., `_c`, `_m`, `_e`)

### Examples
```
CPLX_E3__M_atp__M_glc_c          # Enzyme 3 + ATP + glucose in cytosol
CPLX_E12__M_nad__M_pyr__M_coa_m  # Enzyme 12 + NAD + pyruvate + CoA in mitochondria
CPLX_E5_c                         # Free enzyme 5 in cytosol (rare edge case)
```

### Rationale
- **`CPLX` prefix**: Avoids collision with metabolite IDs; clearly identifies complex type; short and professional
- **Double underscores**: Visually separate enzyme from metabolites; prevent ambiguous parsing
- **Sorted metabolites**: Ensures canonical representation (same complex = same ID)
- **Compartment suffix**: Critical for multi-compartment models; follows BiGG convention

## Elementary Reaction Identifiers

### Ordered Mechanism (Fixed Binding)

#### Substrate Binding Steps
```
<original_rxn_id>_E<enzyme>_SB<step_number>
```
- **`E<enzyme>`**: Enzyme number (e.g., E2, E42)
- **`SB`**: Substrate Binding
- **Example**: `ATPS3m_E2_SB1`, `ATPS3m_E2_SB2`, `ATPS3m_E2_SB3`

#### Catalytic Step
```
<original_rxn_id>_E<enzyme>_CAT
```
- **`CAT`**: Catalysis
- **Example**: `ATPS3m_E2_CAT`

#### Product Release Steps
```
<original_rxn_id>_E<enzyme>_PR<step_number>
```
- **`PR`**: Product Release
- **Example**: `ATPS3m_E2_PR1`, `ATPS3m_E2_PR2`, `ATPS3m_E2_PR3`

### Random Mechanism (All Binding Orders)

#### Substrate Binding Steps
```
<original_rxn_id>_E<enzyme>_SBR<level>_S<substrate_index>_from_<prev_complex>
```
- **`E<enzyme>`**: Enzyme number
- **`SBR`**: Substrate Binding Random mechanism
- **`<level>`**: Binding level (1, 2, 3, ...)
- **`S<index>`**: Which substrate is binding
- **`from_<prev_complex>`**: Previous complex state (for level > 1)
- **Example**: `ATPS3m_E2_SBR1_S1`, `ATPS3m_E2_SBR2_S3_from_1-2`

#### Catalytic Step
```
<original_rxn_id>_E<enzyme>_CAT_RND
```
- **`RND`**: Random mechanism indicator
- **Example**: `ATPS3m_E2_CAT_RND`

#### Product Release Steps
```
<original_rxn_id>_E<enzyme>_PRR<level>_P<product_index>_from_<prev_complex>
```
- **`PRR`**: Product Release Random mechanism
- **Example**: `ATPS3m_E2_PRR1_P1`, `ATPS3m_E2_PRR2_P2_from_1-3`

## Metadata Annotations

All elementary reactions include structured annotations:

```julia
annotations = Dict(
    "elementary_step_type" => ["substrate_binding" | "catalysis" | "product_release"],
    "original_reaction" => [original_reaction_id],
    "mechanism" => ["ordered" | "random"],  # For random mechanism
    "binding_level" => [string(level)],     # For random mechanism
    "substrate" | "product" => [metabolite_id],
    "systematic_name" => [reaction_id]
)
```

All enzyme-substrate complexes include:

```julia
annotations = Dict(
    "sbo" => ["SBO:0000297"],  # Protein-small molecule complex
    "complex_type" => ["enzyme_substrate"],
    "enzyme" => [enzyme_id],
    "bound_metabolites" => [sorted_metabolite_list],
    "systematic_name" => [intermediate_id]
)
```

## Comparison with Previous MATLAB Convention

| Aspect | Old (MATLAB) | New (COCOA.jl) | Benefit |
|--------|-------------|----------------|---------|
| Complex prefix | `E3_` | `CPLX_E3__` | Unambiguous type identification |
| Complex suffix | `_complex` | `_c` (compartment) | Database compatibility |
| Separator | `_` (single) | `__` (double) | Clear logical grouping |
| Reaction suffix | `_e3_s1` | `_E3_SB1` | Self-documenting step type |
| Catalysis | `_s_p_transition` | `_CAT` | Concise and clear |
| Product release | `_p1` | `_PR1` | Explicit step type |

## Benefits Over Ad-hoc Naming

1. **Avoids ambiguity**: Following Pham et al. findings that 83% of mappings can be inconsistent
2. **Enables automation**: Systematic structure allows programmatic parsing
3. **Facilitates model integration**: Compatible with BiGG/MetaNetX standards
4. **Improves traceability**: Clear link between elementary and original reactions
5. **Supports model validation**: Structured annotations enable quality checks

## Implementation Notes

- All metabolite IDs in complexes are **sorted alphabetically** before concatenation
- Compartment information is **always included** (defaults to `c` if unavailable)
- The **double underscore** (`__`) separates major components; single underscore (`_`) separates words within components
- Reaction annotations follow **MIRIAM/COMBINE** standards for semantic annotation

## References

- Pham, N., et al. (2019). "Consistency, Inconsistency, and Ambiguity of Metabolite Names in Biochemical Databases Used for Genome-Scale Metabolic Modelling." *Metabolites* 9(2):28. [DOI:10.3390/metabo9020028](https://doi.org/10.3390/metabo9020028)
- King, Z. A., et al. (2016). "BiGG Models: A platform for integrating, standardizing and sharing genome-scale models." *Nucleic Acids Research* 44(D1):D515-D522.
- Moretti, S., et al. (2016). "MetaNetX/MNXref: unified namespace for metabolites and biochemical reactions in the context of metabolic models." *Nucleic Acids Research* 44(D1):D523-D526.
