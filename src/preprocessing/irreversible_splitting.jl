#=
Irreversible reaction splitting for model preprocessing.

This module converts reversible reactions into irreversible forward/backward pairs,
following COBRA Toolbox conventions while optimized for Julia performance.
=#


export split_into_irreversible

"""
    split_into_irreversible(
        model::CM.Model;
        reactions::Union{Vector{String}, Nothing}=nothing,
        flip_pure_backward::Bool=true
    ) -> CM.Model

Convert reversible reactions to irreversible format by splitting into forward and backward pairs.

Follows COBRA Toolbox conventions:
- Reversible reactions (lb < 0, ub > 0) are split into '_f' (forward) and '_b' (backward)
- Pure backward reactions (lb < 0, ub ≤ 0) are optionally flipped and marked '_r'
- All reaction-associated fields (GPR, annotations, etc.) are properly propagated

The original reaction ID is stored in the `annotations` field under key `"original_reaction"`,
and the paired reaction (for split reactions) is stored under `"paired_reaction"`.

# Arguments
- `model::CM.Model`: Input model to convert

# Keyword Arguments
- `reactions::Union{Vector{String}, Nothing}=nothing`:
    If provided, only split these specific reactions. Otherwise split all reversible reactions.
- `flip_pure_backward::Bool=true`:
    Flip reactions that can only carry negative flux (lb < 0, ub ≤ 0) and mark with '_r'.
    If false, leave them as-is with negative bounds.

# Returns
- `CM.Model`: Model with all specified reversible reactions split

# Reaction Metadata
After splitting, each reaction contains:
- `annotations["original_reaction"]`: Vector with original reaction ID (e.g., `["R_ME1"]`)
- `annotations["paired_reaction"]`: Vector with paired reaction ID for split reactions (e.g., `["R_ME1_b"]`)
- `annotations["split_type"]`: Vector indicating split type: `["forward"]`, `["backward"]`, or `["flipped"]`

# Examples

```julia
# Convert entire model to irreversible
model_irrev = split_into_irreversible(model)

# Convert only specific reactions
model_irrev = split_into_irreversible(
    model,
    reactions=["R1", "R2", "R15"]
)

# Keep backward reactions as negative (don't flip)
model_irrev = split_into_irreversible(
    model,
    flip_pure_backward=false
)

# Query original reaction from split reaction
rxn = model_irrev.reactions["R_ME1_f"]
original_id = rxn.annotations["original_reaction"][1]  # "R_ME1"
paired_id = rxn.annotations["paired_reaction"][1]      # "R_ME1_b"
split_type = rxn.annotations["split_type"][1]          # "forward"

# Find all split reactions from an original
for (rid, rxn) in model_irrev.reactions
    if get(rxn.annotations, "original_reaction", [""]) == ["R_ME1"]
        println("\$rid: \$(rxn.annotations["split_type"][1])")
    end
end
```

# Notes

## Reaction Classification
Reactions are classified into three categories:
1. **Pure forward** (lb ≥ 0, ub > 0): Kept as-is, already irreversible
2. **Pure backward** (lb < 0, ub ≤ 0): Flipped if `flip_pure_backward=true`, marked '_r'
3. **Reversible** (lb < 0, ub > 0): Split into '_f' and '_b' pairs

## Bounds Transformation
For reversible reactions split into forward/backward pairs:
- Forward reaction: `lb_f = max(0, lb)`, `ub_f = max(0, ub)`
- Backward reaction: `lb_b = max(0, -ub)`, `ub_b = max(0, -lb)`

For flipped pure backward reactions:
- Stoichiometry is negated: `S_flipped = -S_original`
- Bounds are flipped: `lb_flipped = -ub_original`, `ub_flipped = -lb_original`
- Objective coefficient is negated to preserve optimization direction

## Field Propagation
All reaction-associated fields are propagated to split reactions:
- `gene_association_dnf`: Copied to both forward and backward
- `annotations`: Copied to both forward and backward
- `notes`: Copied and augmented with split metadata
- `objective_coefficient`: Forward keeps original, backward gets negated value

## Naming Conventions
Following COBRA Toolbox standards:
- Forward reactions: `original_id + "_f"`
- Backward reactions: `original_id + "_b"`
- Flipped reactions: `original_id + "_r"`

"""
function split_into_irreversible(
    model::CM.Model;
    reactions::Union{Vector{String},Nothing}=nothing,
    flip_pure_backward::Bool=true
)

    # Determine which reactions to consider for splitting
    reactions_to_process = isnothing(reactions) ?
                           Set{String}(keys(model.reactions)) :
                           Set{String}(reactions)

    # Pre-scan to classify reactions and allocate efficiently
    n_reactions = length(model.reactions)
    pure_backward_rxns = String[]
    reversible_rxns = String[]

    sizehint!(pure_backward_rxns, n_reactions ÷ 10)  # Estimate ~10% backward
    sizehint!(reversible_rxns, n_reactions ÷ 2)       # Estimate ~50% reversible

    for (rid, rxn) in model.reactions
        if !(rid in reactions_to_process)
            continue
        end

        if rxn.lower_bound < 0 && rxn.upper_bound <= 0
            # Pure backward: lb < 0, ub ≤ 0
            push!(pure_backward_rxns, rid)
        elseif rxn.lower_bound < 0 && rxn.upper_bound > 0
            # Truly reversible: lb < 0, ub > 0
            push!(reversible_rxns, rid)
        end
        # Pure forward (lb ≥ 0) reactions are left unchanged
    end

    n_reversible = length(reversible_rxns)
    n_backward = length(pure_backward_rxns)

    @info "Splitting reactions" reversible = n_reversible pure_backward = n_backward flip = flip_pure_backward

    # Create new model (deep copy to avoid modifying input)
    model_irrev = CM.Model()

    # Copy metabolites and genes (unchanged by splitting)
    model_irrev.metabolites = deepcopy(model.metabolites)
    model_irrev.genes = deepcopy(model.genes)
    model_irrev.couplings = deepcopy(model.couplings)

    # Phase 1: Flip pure backward reactions (if requested)
    if flip_pure_backward && !isempty(pure_backward_rxns)
        for rid in pure_backward_rxns
            rxn = model.reactions[rid]
            new_rid = rid * "_r"

            # Create flipped reaction
            flipped_rxn = CM.Reaction(
                name=rxn.name,
                lower_bound=-rxn.upper_bound,  # Flip bounds
                upper_bound=-rxn.lower_bound,
                stoichiometry=Dict(mid => -coeff for (mid, coeff) in rxn.stoichiometry),  # Negate stoichiometry
                objective_coefficient=-rxn.objective_coefficient,  # Negate objective to preserve direction
                gene_association_dnf=rxn.gene_association_dnf,
                annotations=deepcopy(rxn.annotations),
                notes=deepcopy(rxn.notes)
            )

            # Add metadata tracking the transformation (in annotations for SBML compatibility)
            if !haskey(flipped_rxn.annotations, "original_reaction")
                flipped_rxn.annotations["original_reaction"] = String[]
            end
            if !haskey(flipped_rxn.annotations, "split_type")
                flipped_rxn.annotations["split_type"] = String[]
            end
            push!(flipped_rxn.annotations["original_reaction"], rid)
            push!(flipped_rxn.annotations["split_type"], "flipped")

            model_irrev.reactions[new_rid] = flipped_rxn
        end
    else
        # Keep pure backward reactions as-is (don't flip)
        for rid in pure_backward_rxns
            model_irrev.reactions[rid] = deepcopy(model.reactions[rid])
        end
    end

    # Phase 2: Split reversible reactions into forward/backward pairs
    for rid in reversible_rxns
        rxn = model.reactions[rid]

        # Create forward reaction ID
        fwd_rid = rid * "_f"
        bwd_rid = rid * "_b"

        # Forward reaction: max(0, lb) ≤ v_f ≤ max(0, ub)
        fwd_rxn = CM.Reaction(
            name=rxn.name,
            lower_bound=max(0.0, rxn.lower_bound),
            upper_bound=max(0.0, rxn.upper_bound),
            stoichiometry=deepcopy(rxn.stoichiometry),
            objective_coefficient=rxn.objective_coefficient,
            gene_association_dnf=rxn.gene_association_dnf,
            annotations=deepcopy(rxn.annotations),
            notes=deepcopy(rxn.notes)
        )

        # Backward reaction: max(0, -ub) ≤ v_b ≤ max(0, -lb)
        # Note: Stoichiometry is negated to represent reverse direction
        bwd_rxn = CM.Reaction(
            name=rxn.name,
            lower_bound=max(0.0, -rxn.upper_bound),
            upper_bound=max(0.0, -rxn.lower_bound),
            stoichiometry=Dict(mid => -coeff for (mid, coeff) in rxn.stoichiometry),
            objective_coefficient=-rxn.objective_coefficient,  # Negate to preserve optimization
            gene_association_dnf=rxn.gene_association_dnf,
            annotations=deepcopy(rxn.annotations),
            notes=deepcopy(rxn.notes)
        )

        # Add metadata tracking the split (in annotations for SBML compatibility)
        if !haskey(fwd_rxn.annotations, "original_reaction")
            fwd_rxn.annotations["original_reaction"] = String[]
        end
        if !haskey(fwd_rxn.annotations, "paired_reaction")
            fwd_rxn.annotations["paired_reaction"] = String[]
        end
        if !haskey(fwd_rxn.annotations, "split_type")
            fwd_rxn.annotations["split_type"] = String[]
        end
        push!(fwd_rxn.annotations["original_reaction"], rid)
        push!(fwd_rxn.annotations["paired_reaction"], bwd_rid)
        push!(fwd_rxn.annotations["split_type"], "forward")

        if !haskey(bwd_rxn.annotations, "original_reaction")
            bwd_rxn.annotations["original_reaction"] = String[]
        end
        if !haskey(bwd_rxn.annotations, "paired_reaction")
            bwd_rxn.annotations["paired_reaction"] = String[]
        end
        if !haskey(bwd_rxn.annotations, "split_type")
            bwd_rxn.annotations["split_type"] = String[]
        end
        push!(bwd_rxn.annotations["original_reaction"], rid)
        push!(bwd_rxn.annotations["paired_reaction"], fwd_rid)
        push!(bwd_rxn.annotations["split_type"], "backward")

        model_irrev.reactions[fwd_rid] = fwd_rxn
        model_irrev.reactions[bwd_rid] = bwd_rxn
    end

    # Phase 3: Copy unchanged reactions (pure forward, or not in specific_reactions list)
    reversible_set = Set(reversible_rxns)
    backward_set = Set(pure_backward_rxns)

    for (rid, rxn) in model.reactions
        if !(rid in reversible_set) && !(rid in backward_set)
            # This reaction wasn't processed - copy as-is
            model_irrev.reactions[rid] = deepcopy(rxn)
        end
    end

    n_final = length(model_irrev.reactions)
    @info "Irreversible conversion complete" original = n_reactions final = n_final added = n_final - n_reactions
    return model_irrev
end
