"""
Mass action kinetics validation for elementary step models.

This module provides validation functions to ensure that the elementary step
splitting produces models compatible with mass action kinetics.
"""

import AbstractFBCModels.CanonicalModel as CM
import AbstractFBCModels as A
import Logging

export validate_mass_action_kinetics, check_enzyme_conservation, 
       validate_elementary_steps, analyze_reaction_complexity

"""
    validate_mass_action_kinetics(model::A.AbstractFBCModel; verbose::Bool=true)

Comprehensive validation that a model is suitable for mass action kinetics.

Checks:
1. All reactions are elementary (≤ 2 reactants per step)
2. Enzyme conservation is maintained
3. Intermediate complexes are properly balanced
4. No unrealistic stoichiometry

Returns (is_valid::Bool, violations::Vector{String})
"""
function validate_mass_action_kinetics(model::A.AbstractFBCModel; verbose::Bool=true)
    violations = String[]
    
    # Convert to canonical for easier analysis
    canon_model = convert(CM.Model, model)
    
    # Check 1: Elementary step structure
    elementary_violations = validate_elementary_steps(canon_model)
    append!(violations, elementary_violations)
    
    # Check 2: Enzyme conservation
    enzyme_violations = check_enzyme_conservation(canon_model)
    append!(violations, enzyme_violations)
    
    # Check 3: Reaction complexity analysis
    complexity_violations = analyze_reaction_complexity(canon_model)
    append!(violations, complexity_violations)
    
    is_valid = isempty(violations)
    
    if verbose
        if is_valid
            @info "✓ Model passes mass action kinetics validation"
        else
            @warn "⚠ Model has $(length(violations)) mass action violations:"
            for violation in violations
                @warn "  - $violation"
            end
        end
    end
    
    return is_valid, violations
end

"""
    validate_elementary_steps(model::CM.Model)

Check that all reactions follow elementary step principles:
- At most 2 reactants (for bimolecular elementary reactions)
- Simple stoichiometry (typically coefficients of 1)
- Proper enzyme involvement
"""
function validate_elementary_steps(model::CM.Model)
    violations = String[]
    
    for (rid, rxn) in model.reactions
        # Count reactants and products
        reactants = [(mid, abs(coeff)) for (mid, coeff) in rxn.stoichiometry if coeff < 0]
        products = [(mid, coeff) for (mid, coeff) in rxn.stoichiometry if coeff > 0]
        
        # Check reactant count (elementary reactions should have ≤ 2 reactants)
        if length(reactants) > 2
            push!(violations, "Reaction $rid has $(length(reactants)) reactants (should be ≤ 2 for elementary steps)")
        end
        
        # Check for unusual stoichiometry
        unusual_stoich = false
        for (mid, coeff) in [reactants; products]
            if coeff > 3.0  # Allow some flexibility for biological stoichiometry
                unusual_stoich = true
                break
            end
        end
        
        if unusual_stoich
            push!(violations, "Reaction $rid has unusual stoichiometry (may not be elementary)")
        end
        
        # Check enzyme involvement for non-transport reactions
        if !is_transport_like_reaction(rid, reactants, products)
            has_enzyme = any(contains(mid, "ENZ_") for (mid, _) in [reactants; products])
            if !has_enzyme && length(reactants) > 1
                push!(violations, "Complex reaction $rid lacks enzyme involvement")
            end
        end
    end
    
    return violations
end

"""
    check_enzyme_conservation(model::CM.Model)

Verify that enzymes are properly conserved in the reaction network.
For each enzyme, total enzyme = free enzyme + sum of all enzyme complexes.
"""
function check_enzyme_conservation(model::CM.Model)
    violations = String[]
    
    # Identify all enzymes and their complexes
    enzymes = Set{String}()
    enzyme_complexes = Dict{String, Vector{String}}()  # enzyme -> complexes involving it
    
    for (mid, met) in model.metabolites
        if startswith(mid, "ENZ_") && !contains(mid, "_complex")
            # Free enzyme
            push!(enzymes, mid)
            enzyme_complexes[mid] = String[]
        elseif contains(mid, "_complex") && contains(mid, "ENZ_")
            # Enzyme complex - extract the enzyme
            enzyme_match = match(r"(ENZ_[^_]+)", mid)
            if !isnothing(enzyme_match)
                enzyme_id = enzyme_match.captures[1]
                if !haskey(enzyme_complexes, enzyme_id)
                    enzyme_complexes[enzyme_id] = String[]
                end
                push!(enzyme_complexes[enzyme_id], mid)
            end
        end
    end
    
    # Check conservation for each enzyme
    for enzyme_id in enzymes
        complexes = get(enzyme_complexes, enzyme_id, String[])
        
        # Get stoichiometric matrix for this enzyme and its complexes
        enzyme_mets = [enzyme_id; complexes]
        
        # Check if enzyme appears in any reactions
        enzyme_reactions = String[]
        for (rid, rxn) in model.reactions
            if any(haskey(rxn.stoichiometry, met) for met in enzyme_mets)
                push!(enzyme_reactions, rid)
            end
        end
        
        # For enzymatic pathways, we expect enzyme conservation
        if length(enzyme_reactions) > 1
            # Simple check: enzyme should sum to zero across related reactions
            total_enzyme_balance = 0.0
            for (rid, rxn) in model.reactions
                if rid in enzyme_reactions
                    for met in enzyme_mets
                        total_enzyme_balance += get(rxn.stoichiometry, met, 0.0)
                    end
                end
            end
            
            # Note: This is a simplified check. Perfect enzyme conservation
            # requires more sophisticated analysis of reaction pathways
            if abs(total_enzyme_balance) > 1e-10
                push!(violations, "Enzyme $enzyme_id may not be properly conserved (net balance: $total_enzyme_balance)")
            end
        end
    end
    
    return violations
end

"""
    analyze_reaction_complexity(model::CM.Model)

Analyze reaction complexity and identify potential issues for mass action kinetics.
"""
function analyze_reaction_complexity(model::CM.Model)
    violations = String[]
    
    # Count different types of reactions
    elementary_count = 0
    complex_count = 0
    transport_count = 0
    
    for (rid, rxn) in model.reactions
        reactants = [mid for (mid, coeff) in rxn.stoichiometry if coeff < 0]
        products = [mid for (mid, coeff) in rxn.stoichiometry if coeff > 0]
        
        if is_transport_like_reaction(rid, [(mid, 1.0) for mid in reactants], 
                                     [(mid, 1.0) for mid in products])
            transport_count += 1
        elseif length(reactants) <= 2 && length(products) <= 2
            elementary_count += 1
        else
            complex_count += 1
            # Don't warn about individual complex reactions - this is normal in hybrid models
            # The upstream algorithm keeps many reactions unexpanded by design
        end
    end
    
    total_reactions = length(model.reactions)
    elementary_fraction = elementary_count / total_reactions
    
    @info """Reaction complexity analysis:
    - Elementary reactions (≤2 reactants, ≤2 products): $elementary_count ($(round(elementary_fraction*100, digits=1))%)
    - Complex reactions: $complex_count
    - Transport-like reactions: $transport_count
    - Total reactions: $total_reactions"""
    
    # Note: In hybrid models (like upstream algorithm), having unexpanded reactions is normal
    # Only warn if very few reactions are elementary (indicating potential issues)
    if elementary_fraction < 0.1  # Very conservative threshold
        push!(violations, "Only $(round(elementary_fraction*100, digits=1))% of reactions are elementary - this severely limits mass action kinetics applicability")
    end
    
    # Always provide summary info but not as warning
    @info "Hybrid model composition: $(round(elementary_fraction*100, digits=1))% elementary reactions suitable for mass action kinetics"
    
    return violations
end

"""
    is_transport_like_reaction(rid::String, reactants, products)

Heuristic to identify transport or exchange reactions that don't need enzyme kinetics.
"""
function is_transport_like_reaction(rid::String, reactants, products)
    # Check reaction ID patterns
    if startswith(rid, "EX_") || startswith(rid, "DM_") || 
       contains(rid, "transport") || contains(rid, "_exchange") ||
       contains(rid, "_demand") || contains(rid, "sink_")
        return true
    end
    
    # Check if same metabolite appears in different compartments
    if length(reactants) == 1 && length(products) == 1
        reactant_base = replace(reactants[1][1], r"_[a-z]$" => "")
        product_base = replace(products[1][1], r"_[a-z]$" => "")
        if reactant_base == product_base
            return true  # Same metabolite, different compartments
        end
    end
    
    return false
end