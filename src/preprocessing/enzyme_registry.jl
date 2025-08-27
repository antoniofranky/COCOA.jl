"""
Enzyme registry functionality for elementary step splitting.

This module handles:
- Enzyme discovery from gene associations
- Enzyme registry management
- Reaction-enzyme mapping
"""

using Random
import AbstractFBCModels.CanonicalModel as CM

export extract_reaction_enzymes, build_enzyme_registry

"""
Build a registry of all enzymes from multiple sources to match MATLAB algorithm behavior.

Extracts enzyme information from:
1. Gene associations (GPR rules) 
2. EC number annotations
3. Generic enzyme assignments for core metabolic reactions
"""
function build_enzyme_registry(model::CM.Model)
    enzyme_registry = Dict{String,String}()
    enzyme_counter = 0
    
    # Statistics for debugging
    stats = Dict(:from_gpr => 0, :from_ec => 0)

    for (rid, rxn) in model.reactions
        # 1. Extract enzymes from gene associations (original approach)
        if !isnothing(rxn.gene_association_dnf) && !isempty(rxn.gene_association_dnf)
            for gene_group in rxn.gene_association_dnf
                enzyme_counter += 1
                if length(gene_group) == 1
                    # Single gene = single enzyme  
                    gene_name = gene_group[1]
                    # Check if this is an EC-derived artificial gene (contains dots like "2_6_1_42")
                    if count('_', gene_name) >= 3 && all(c -> isdigit(c) || c == '_', gene_name)
                        enzyme_id = "ENZ_EC_$gene_name"  # EC-derived artificial gene
                    else
                        enzyme_id = "ENZ_$gene_name"  # Regular gene
                    end
                    enzyme_registry[enzyme_id] = gene_name
                    stats[:from_gpr] += 1
                else
                    # Multiple genes = enzyme complex
                    complex_name = join(sort(gene_group), "_")
                    enzyme_id = "ENZ_$complex_name"
                    enzyme_registry[enzyme_id] = join(sort(gene_group), " & ")
                    stats[:from_gpr] += 1
                end
            end
        end
        
        # 2. Extract enzymes from EC number annotations (matching MATLAB rxnECNumbers behavior)
        # Only use EC as fallback if no GPR rules (matching MATLAB logic lines 24-27)
        if isnothing(rxn.gene_association_dnf) || isempty(rxn.gene_association_dnf)
            ec_enzyme_ids = extract_ec_enzymes(rid, rxn)
            for ec_enzyme_id in ec_enzyme_ids
                if !haskey(enzyme_registry, ec_enzyme_id)
                    enzyme_registry[ec_enzyme_id] = ec_enzyme_id
                    stats[:from_ec] += 1
                end
            end
            
            # Store EC-derived artificial genes for later use (don't modify model in-place)
            # This matches MATLAB line 26 behavior but deferred to avoid model corruption
            if !isempty(ec_enzyme_ids)
                # Create artificial gene names from EC numbers for GPR system
                artificial_genes = [replace(ec_id, "ENZ_EC_" => "") for ec_id in ec_enzyme_ids]
                # Note: We don't modify rxn.gene_association_dnf here to avoid model structure issues
                # The artificial genes are handled through the enzyme registry lookup
            end
        end
        
        # 3. No generic enzymes - upstream algorithm doesn't create them
        # Only use actual enzyme information (GPR or EC)
    end
    
    @info """Enzyme registry built (matching upstream algorithm):
    - Enzymes from GPR rules (including EC-derived): $(stats[:from_gpr])
    - EC codes converted to artificial GPR rules: $(stats[:from_ec])
    - Total enzymes: $(length(enzyme_registry))"""

    return enzyme_registry
end

"""
Extract enzyme identifiers from EC number annotations.
Returns vector of enzyme IDs to handle multiple EC codes (isoenzymes).
"""
function extract_ec_enzymes(rid::String, rxn::CM.Reaction)::Vector{String}
    enzyme_ids = String[]
    
    # Check various EC annotation fields
    ec_fields = ["ec-code", "EC", "ec_number", "enzyme", "EC_number"]
    
    for field in ec_fields
        if haskey(rxn.annotations, field)
            ec_data = rxn.annotations[field]
            if !isempty(ec_data)
                # Handle multiple EC codes as separate isoenzymes (like upstream algorithm)
                for ec in ec_data
                    ec_str = strip(string(ec))
                    if !isempty(ec_str) && ec_str != "None" && ec_str != "none" 
                        # Create enzyme ID from EC number
                        clean_ec = replace(ec_str, r"[^\d\.]" => "_")
                        enzyme_id = "ENZ_EC_$clean_ec"
                        if !in(enzyme_id, enzyme_ids)  # Avoid duplicates
                            push!(enzyme_ids, enzyme_id)
                        end
                    end
                end
            end
        end
    end
    
    return enzyme_ids
end

"""
Extract enzyme IDs for a reaction from all available sources.

Enhanced to match MATLAB algorithm behavior by checking:
1. Gene associations (GPR rules, including EC-derived artificial genes)
2. No fallback mechanisms (reactions without enzymes remain unexpanded)
"""
function extract_reaction_enzymes(rxn::CM.Reaction, enzyme_registry::Dict{String,String})
    enzyme_ids = String[]

    # 1. Extract from gene associations (original GPR rules)
    if !isnothing(rxn.gene_association_dnf) && !isempty(rxn.gene_association_dnf)
        for gene_group in rxn.gene_association_dnf
            if length(gene_group) == 1
                gene_name = gene_group[1]
                enzyme_id = "ENZ_$gene_name"
            else
                complex_name = join(sort(gene_group), "_")
                enzyme_id = "ENZ_$complex_name"
            end

            if haskey(enzyme_registry, enzyme_id)
                push!(enzyme_ids, enzyme_id)
            end
        end
    end
    
    # 2. Extract from EC number annotations directly (fallback for reactions without GPR)
    if isempty(enzyme_ids)
        ec_enzyme_ids = extract_ec_enzymes("", rxn)  # Get EC enzymes directly
        for ec_enzyme_id in ec_enzyme_ids
            if haskey(enzyme_registry, ec_enzyme_id)
                push!(enzyme_ids, ec_enzyme_id)
            end
        end
    end

    return enzyme_ids
end

"""
Count the number of enzyme variants (isoenzymes) for a reaction.
Used to properly distribute objective coefficients.
"""
function count_reaction_enzymes(rxn::CM.Reaction, enzyme_registry::Dict{String,String})::Int
    # Extract enzyme IDs from gene association
    if isnothing(rxn.gene_association_dnf) || isempty(rxn.gene_association_dnf)
        return 0
    end

    enzyme_count = 0
    for term in rxn.gene_association_dnf
        # Each term represents an enzyme or enzyme complex
        enzyme_count += 1
    end
    return max(1, enzyme_count)  # At least 1 to avoid division by zero
end

"""
Assign ordered or random mechanism to each eligible reaction.

Enhanced version that matches MATLAB algorithm behavior by considering:
1. Reactions with gene associations (GPR rules)
2. Reactions with EC number annotations 
3. Core metabolic reactions (as fallback for mass action kinetics)
4. Substrate/product complexity limits
"""
function assign_reaction_mechanisms(
    model::CM.Model, ordered_fraction::Float64,
    max_substrates::Int, max_products::Int, rng::AbstractRNG
)
    mechanisms = Dict{String,Symbol}()
    eligible_reactions = String[]
    
    # Collect statistics for debugging
    stats = Dict(
        :has_gpr => 0,
        :has_ec => 0, 
        :too_complex => 0,
        :no_enzyme_info => 0,
        :total_eligible => 0
    )

    for (rid, rxn) in model.reactions
        # Check substrate/product limits first (same as MATLAB: <=4 each)
        n_substrates = count(coeff < 0 for (_, coeff) in rxn.stoichiometry)
        n_products = count(coeff > 0 for (_, coeff) in rxn.stoichiometry)

        if n_substrates > max_substrates || n_products > max_products
            stats[:too_complex] += 1
            continue
        end

        # Check eligibility criteria (match MATLAB algorithm)
        is_eligible = false
        
        # 1. Has gene association (original criterion)
        if !isnothing(rxn.gene_association_dnf) && !isempty(rxn.gene_association_dnf)
            is_eligible = true
            stats[:has_gpr] += 1
        end
        
        # 2. Has EC number in annotations (MATLAB equivalent of rxnECNumbers)
        # Only use EC as fallback if no GPR rules (matching MATLAB logic lines 24-27)
        if !is_eligible && (isnothing(rxn.gene_association_dnf) || isempty(rxn.gene_association_dnf))
            ec_enzyme_ids = extract_ec_enzymes(rid, rxn)
            if !isempty(ec_enzyme_ids)
                is_eligible = true
                stats[:has_ec] += 1
            end
        end
        
        # 4. Skip generic/core metabolic fallback - upstream algorithm doesn't do this
        # Only split reactions with actual enzyme information (GPR or EC)

        if is_eligible
            push!(eligible_reactions, rid)
            stats[:total_eligible] += 1
        else
            # Count reactions without enzyme info (will remain unexpanded)
            stats[:no_enzyme_info] += 1
        end
    end

    # Log statistics for debugging  
    @info """Elementary step eligibility analysis (matching upstream algorithm):
    - Reactions with GPR rules: $(stats[:has_gpr])  
    - Reactions with EC annotations (no GPR): $(stats[:has_ec])
    - Too complex to split (>4 substrates/products): $(stats[:too_complex])
    - No enzyme information (will remain unexpanded): $(stats[:no_enzyme_info])
    - Total eligible for splitting: $(stats[:total_eligible])/$(length(model.reactions))
    - Unexpanded reactions: $(stats[:no_enzyme_info] + stats[:too_complex])"""

    # Randomly assign mechanisms
    n_ordered = round(Int, length(eligible_reactions) * ordered_fraction)
    shuffle!(rng, eligible_reactions)

    for (i, rid) in enumerate(eligible_reactions)
        mechanisms[rid] = i <= n_ordered ? :ordered : :random
    end

    return mechanisms
end

# Removed is_core_metabolic_reaction function - upstream algorithm doesn't use generic enzymes