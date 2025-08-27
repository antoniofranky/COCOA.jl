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
    stats = Dict(:from_gpr => 0, :from_ec => 0, :generic => 0)

    for (rid, rxn) in model.reactions
        # 1. Extract enzymes from gene associations (original approach)
        if !isnothing(rxn.gene_association_dnf) && !isempty(rxn.gene_association_dnf)
            for gene_group in rxn.gene_association_dnf
                enzyme_counter += 1
                if length(gene_group) == 1
                    # Single gene = single enzyme
                    enzyme_id = "ENZ_$(gene_group[1])"
                    enzyme_registry[enzyme_id] = gene_group[1]
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
        ec_enzyme_id = extract_ec_enzyme(rid, rxn)
        if !isnothing(ec_enzyme_id) && !haskey(enzyme_registry, ec_enzyme_id)
            enzyme_registry[ec_enzyme_id] = ec_enzyme_id
            stats[:from_ec] += 1
        end
        
        # 3. Generic enzyme for core metabolic reactions (fallback)
        if isnothing(rxn.gene_association_dnf) && isnothing(ec_enzyme_id)
            n_substrates = count(coeff < 0 for (_, coeff) in rxn.stoichiometry)
            n_products = count(coeff > 0 for (_, coeff) in rxn.stoichiometry)
            
            if is_core_metabolic_reaction(rid, rxn, n_substrates, n_products)
                generic_enzyme_id = "ENZ_GENERIC_$rid"
                enzyme_registry[generic_enzyme_id] = "Generic enzyme for $rid"
                stats[:generic] += 1
            end
        end
    end
    
    @info """Enzyme registry built:
    - Enzymes from GPR rules: $(stats[:from_gpr])
    - Enzymes from EC annotations: $(stats[:from_ec]) 
    - Generic enzymes: $(stats[:generic])
    - Total enzymes: $(length(enzyme_registry))"""

    return enzyme_registry
end

"""
Extract enzyme identifier from EC number annotations.
"""
function extract_ec_enzyme(rid::String, rxn::CM.Reaction)::Union{String, Nothing}
    # Check various EC annotation fields
    ec_fields = ["ec-code", "EC", "ec_number", "enzyme", "EC_number"]
    
    for field in ec_fields
        if haskey(rxn.annotations, field)
            ec_data = rxn.annotations[field]
            if !isempty(ec_data)
                # Take the first non-empty EC number
                for ec in ec_data
                    ec_str = strip(string(ec))
                    if !isempty(ec_str) && ec_str != "None" && ec_str != "none" 
                        # Create enzyme ID from EC number
                        clean_ec = replace(ec_str, r"[^\d\.]" => "_")
                        return "ENZ_EC_$clean_ec"
                    end
                end
            end
        end
    end
    
    return nothing
end

"""
Extract enzyme IDs for a reaction from all available sources.

Enhanced to match MATLAB algorithm behavior by checking:
1. Gene associations (GPR rules)
2. EC number annotations 
3. Generic enzymes for core metabolic reactions
"""
function extract_reaction_enzymes(rxn::CM.Reaction, enzyme_registry::Dict{String,String}, 
                                 rid::String="")
    enzyme_ids = String[]

    # 1. Extract from gene associations (original approach)
    if !isnothing(rxn.gene_association_dnf) && !isempty(rxn.gene_association_dnf)
        for gene_group in rxn.gene_association_dnf
            if length(gene_group) == 1
                enzyme_id = "ENZ_$(gene_group[1])"
            else
                complex_name = join(sort(gene_group), "_")
                enzyme_id = "ENZ_$complex_name"
            end

            if haskey(enzyme_registry, enzyme_id)
                push!(enzyme_ids, enzyme_id)
            end
        end
    end
    
    # 2. Extract from EC number annotations
    if isempty(enzyme_ids)  # Only if no GPR enzymes found
        ec_enzyme_id = extract_ec_enzyme(rid, rxn)
        if !isnothing(ec_enzyme_id) && haskey(enzyme_registry, ec_enzyme_id)
            push!(enzyme_ids, ec_enzyme_id)
        end
    end
    
    # 3. Generic enzyme fallback
    if isempty(enzyme_ids)  # Only if no other enzymes found
        generic_enzyme_id = "ENZ_GENERIC_$rid"
        if haskey(enzyme_registry, generic_enzyme_id)
            push!(enzyme_ids, generic_enzyme_id)
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
        :is_core_metabolic => 0,
        :too_complex => 0,
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
        if !is_eligible && haskey(rxn.annotations, "ec-code")
            ec_codes = rxn.annotations["ec-code"]
            if !isempty(ec_codes) && any(ec -> !isempty(strip(ec)), ec_codes)
                is_eligible = true
                stats[:has_ec] += 1
            end
        end
        
        # 3. Alternative EC annotations
        if !is_eligible
            for ec_field in ["EC", "ec_number", "enzyme", "EC_number"]
                if haskey(rxn.annotations, ec_field)
                    ec_data = rxn.annotations[ec_field]
                    if !isempty(ec_data) && any(ec -> !isempty(strip(string(ec))), ec_data)
                        is_eligible = true
                        stats[:has_ec] += 1
                        break
                    end
                end
            end
        end
        
        # 4. Core metabolic reactions (fallback for mass action kinetics)
        # Include reactions that have reasonable complexity for enzyme kinetics
        if !is_eligible && is_core_metabolic_reaction(rid, rxn, n_substrates, n_products)
            is_eligible = true
            stats[:is_core_metabolic] += 1
        end

        if is_eligible
            push!(eligible_reactions, rid)
            stats[:total_eligible] += 1
        end
    end

    # Log statistics for debugging
    @info """Elementary step eligibility analysis:
    - Reactions with GPR rules: $(stats[:has_gpr])  
    - Reactions with EC annotations: $(stats[:has_ec])
    - Core metabolic reactions: $(stats[:is_core_metabolic])
    - Too complex to split: $(stats[:too_complex])
    - Total eligible for splitting: $(stats[:total_eligible])/$(length(model.reactions))"""

    # Randomly assign mechanisms
    n_ordered = round(Int, length(eligible_reactions) * ordered_fraction)
    shuffle!(rng, eligible_reactions)

    for (i, rid) in enumerate(eligible_reactions)
        mechanisms[rid] = i <= n_ordered ? :ordered : :random
    end

    return mechanisms
end

"""
Determine if a reaction should be considered core metabolic and eligible for splitting.

This provides a fallback mechanism to ensure sufficient reactions are split for 
mass action kinetics, matching the upstream algorithm's permissive approach.
"""
function is_core_metabolic_reaction(rid::String, rxn::CM.Reaction, 
                                   n_substrates::Int, n_products::Int)::Bool
    # Skip if reaction is too simple (likely transport/exchange)
    if n_substrates + n_products <= 1
        return false
    end
    
    # Skip if reaction looks like transport (same metabolite different compartments)
    metabolite_names = Set{String}()
    for (met_id, _) in rxn.stoichiometry
        # Extract base metabolite name (remove compartment suffix)
        base_name = replace(met_id, r"_[a-z]$" => "")
        push!(metabolite_names, base_name)
    end
    
    # If only one unique metabolite name, likely transport
    if length(metabolite_names) == 1
        return false
    end
    
    # Skip exchange reactions (typically have "EX_" prefix or similar patterns)
    if startswith(rid, "EX_") || startswith(rid, "DM_") || startswith(rid, "sink_") || 
       contains(rid, "_exchange") || contains(rid, "_demand")
        return false
    end
    
    # Include reactions with moderate complexity that could benefit from enzyme kinetics
    # This matches MATLAB's inclusive approach for metabolic reactions
    return (n_substrates >= 1 && n_products >= 1 && 
            n_substrates <= 3 && n_products <= 3)
end