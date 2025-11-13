"""
EXACT line-by-line replication of split_into_elementary_rxns_v1.m

This follows the MATLAB code structure precisely to ensure identical results.
Use for validation only - not optimized for Julia performance.

NOTE: Uses Dict-based complex registry instead of MATLAB's matrix approach for efficiency.

## EC Code Fallback

Reactions without GPR rules but with EC code annotations (e.g., "ec-code" => ["1.7.1.3"])
will use the EC code as an artificial gene identifier for splitting. This matches MATLAB's
behavior (lines 23-27 in split_into_elementary_rxns_v1.m).

EC (Enzyme Commission) numbers classify enzymes by the reactions they catalyze:
- Format: EC X.Y.Z.W (e.g., EC 1.7.1.3 = nitrate reductase)
- Used when enzymatic function is known but specific genes are not annotated
"""

# Import mechanism functions
include("mechanisms.jl")

"""
    extract_ec_number(rxn::CM.Reaction) -> Union{String, Nothing}

Extract EC number from reaction annotations or notes field.

Checks in order:
1. annotations["ec-code"] - structured annotation
2. notes field - parses XML/HTML for "EC Number: X.Y.Z.W"

Returns the EC number string or nothing if not found or invalid (e.g., "--").
"""
function extract_ec_number(rxn::CM.Reaction)
    # First try annotations (structured data)
    if haskey(rxn.annotations, "ec-code") && !isempty(rxn.annotations["ec-code"])
        ec = rxn.annotations["ec-code"][1]
        # Filter out invalid placeholders
        if ec != "--" && ec != "-" && !isempty(ec)
            return ec
        end
    end

    # Try parsing from notes field (legacy format)
    if haskey(rxn.notes, "") && !isempty(rxn.notes[""])
        notes_text = rxn.notes[""][1]
        # Look for pattern: "EC Number: X.Y.Z.W" or "EC Number: --"
        m = match(r"EC Number:\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", notes_text)
        if !isnothing(m)
            return m.captures[1]
        end
    end

    return nothing
end
"""
    split_into_elementary(
        model::CM.Model;
        random::Float64=0.0,
        seed::Union{Int,Nothing}=nothing
    ) -> CM.Model
Split reactions into elementary steps using ordered or random mechanisms.
"""
function split_into_elementary(
    model::CM.Model;
    random::Float64=0.0,
    seed::Union{Int,Nothing}=nothing
)
    @assert 0.0 <= random <= 1.0 "random must be between 0.0 and 1.0"

    # Initialize RNG for reproducible mechanism assignment
    rng = isnothing(seed) ? Random.default_rng() : StableRNGs.StableRNG(seed)

    # MATLAB line 23-27: Use EC codes as fallback for missing GPR rules
    # Create a modified model where EC codes fill in for empty GPR rules
    reactions_with_fallback = Dict{String,Vector{Vector{String}}}()
    for (rid, rxn) in model.reactions
        gpr = rxn.gene_association_dnf

        # Check if GPR is empty but EC code exists
        has_empty_gpr = isnothing(gpr) || isempty(gpr) || all(isempty(g) for g in gpr)

        if has_empty_gpr
            # Try to extract EC number from annotations or notes
            ec_code = extract_ec_number(rxn)

            if !isnothing(ec_code)
                # Use EC code as artificial gene for splitting
                reactions_with_fallback[rid] = [[ec_code]]
                @info "Using EC code fallback for reaction $rid" ec_code = ec_code
            end
        else
            reactions_with_fallback[rid] = gpr
        end
        # If neither GPR nor EC code exists, reaction won't be split (matching MATLAB)
    end

    # MATLAB line 29-34: Build enzyme list from GPR rules (including EC code fallbacks)
    # FIX: Only include enzymes from reactions that will actually be split
    enzyme_list = String[]
    enzyme_to_reactions = Dict{String,Vector{String}}()  # Track which reactions use each enzyme

    for (rid, gpr) in reactions_with_fallback
        rxn = model.reactions[rid]
        substrate_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff < 0]
        product_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff > 0]

        # Only process enzymes from reactions that will be split (≤4 substrates AND ≤4 products)
        if length(substrate_ids) <= 4 && length(product_ids) <= 4
            for gene_group in gpr
                enzyme_name = length(gene_group) == 1 ? gene_group[1] : join(sort(gene_group), " & ")
                if !(enzyme_name in enzyme_list)
                    push!(enzyme_list, enzyme_name)
                    enzyme_to_reactions[enzyme_name] = String[]
                end
                push!(enzyme_to_reactions[enzyme_name], rid)
            end
        end
    end

    # Create ordered list of metabolite IDs (MATLAB uses array indices)
    met_ids = sort(collect(keys(model.metabolites)))
    NMET = length(met_ids)
    NENZ = length(enzyme_list)

    println("MATLAB-exact: $NMET metabolites, $NENZ enzymes, $(length(model.reactions)) reactions")

    # MATLAB line 40-55: Initialize elementary model
    elem_model = CM.Model()

    # Copy metabolites
    for mid in met_ids
        elem_model.metabolites[mid] = deepcopy(model.metabolites[mid])
    end

    # Add enzyme metabolites (MATLAB line 44-46: E1, E2, ...)
    for e_idx in 1:NENZ
        enz_id = "E$e_idx"
        elem_model.metabolites[enz_id] = CM.Metabolite(
            name="Enzyme: $(enzyme_list[e_idx])",
            compartment="c"
        )
    end

    # Track intermediates: Map sorted metabolite list -> intermediate metabolite ID
    # mechanisms.jl uses Vector{String} as key (sorted list of metabolites + enzyme)
    intermediate_registry = Dict{Vector{String},String}()

    # Compute global bounds (MATLAB: min([-1000, min(model.lb)]))
    all_lbs = [rxn.lower_bound for (_, rxn) in model.reactions]
    all_ubs = [rxn.upper_bound for (_, rxn) in model.reactions]
    reversible_lb = min(-1000.0, minimum(all_lbs))
    reversible_ub = max(1000.0, maximum(all_ubs))

    println("Global bounds: [$reversible_lb, $reversible_ub]")

    # Pre-compute mechanism assignment for reactions with GPR (including EC code fallbacks)
    reactions_to_expand = String[]
    for (rid, gpr) in reactions_with_fallback
        rxn = model.reactions[rid]
        substrate_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff < 0]
        product_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff > 0]

        if length(substrate_ids) <= 4 && length(product_ids) <= 4
            push!(reactions_to_expand, rid)
        end
    end

    # Assign random mechanism to a fraction of expandable reactions
    n_random = round(Int, length(reactions_to_expand) * random)
    random_rxns = Set(Random.shuffle(rng, reactions_to_expand)[1:n_random])

    @info "Mechanism assignment" total_expandable = length(reactions_to_expand) random = n_random ordered = length(reactions_to_expand) - n_random

    # MATLAB line 57: Process each reaction
    for (rid, rxn) in model.reactions
        # MATLAB line 59-60: Get substrate/product indices
        substrate_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff < 0]
        product_ids = [mid for (mid, coeff) in rxn.stoichiometry if coeff > 0]

        # MATLAB line 62: Check if should remain unexpanded
        # Use fallback GPR if available, otherwise check original
        has_gpr_or_ec = haskey(reactions_with_fallback, rid)

        if !has_gpr_or_ec || length(substrate_ids) > 4 || length(product_ids) > 4
            # MATLAB line 63-70: Keep original reaction
            # FIX: Keep R_ prefix for SBML compatibility (prevents objective reference issues)
            rid_clean = startswith(rid, "R_") ? rid : "R_" * rid
            elem_model.reactions[rid_clean] = deepcopy(rxn)
        else
            # MATLAB line 77-78: Extract enzyme indices for this reaction
            # Use fallback GPR (includes EC codes) instead of original GPR
            rxn_gpr = reactions_with_fallback[rid]
            rxn_enzymes = Int[]
            for gene_group in rxn_gpr
                enz_name = length(gene_group) == 1 ? gene_group[1] : join(sort(gene_group), " & ")
                enz_idx = findfirst(==(enz_name), enzyme_list)
                if !isnothing(enz_idx)
                    push!(rxn_enzymes, enz_idx)
                end
            end

            # Determine mechanism for this reaction
            use_random = rid in random_rxns

            # Convert substrate/product IDs to (id, coefficient) tuples for mechanisms.jl
            substrates = [(mid, -coeff) for (mid, coeff) in rxn.stoichiometry if coeff < 0]
            products = [(mid, coeff) for (mid, coeff) in rxn.stoichiometry if coeff > 0]

            # FIX: Keep R_ prefix for SBML compatibility (prevents objective reference issues)
            rid_clean = startswith(rid, "R_") ? rid : "R_" * rid

            if !use_random
                # MATLAB line 72-215: Ordered mechanism
                for e_idx in rxn_enzymes
                    enzyme_id = "E$e_idx"
                    add_ordered_reactions!(
                        elem_model, rid_clean, rxn, enzyme_id,
                        substrates, products,
                        intermediate_registry,
                        model  # Pass original model for compartment lookup
                    )
                end
            else
                # MATLAB line 216-524: Random mechanism (generates ALL binding orders, deterministic)
                for e_idx in rxn_enzymes
                    enzyme_id = "E$e_idx"
                    add_random_reactions!(
                        elem_model, rid_clean, rxn, enzyme_id,
                        substrates, products,
                        intermediate_registry,
                        model  # Pass original model for compartment lookup
                    )
                end
            end
        end
    end

    # Copy genes
    elem_model.genes = deepcopy(model.genes)

    println("Created $(length(elem_model.reactions)) elementary reactions, $(length(intermediate_registry)) intermediates")


    return elem_model
end
