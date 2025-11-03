"""
Detailed analysis of reaction count differences between Julia and MATLAB.

This script identifies which specific base reactions account for the discrepancy.
"""

println("="^80)
println("DETAILED REACTION DIFFERENCE ANALYSIS")
println("="^80)
println()

# Load reaction lists
julia_rxns_file = "julia_reactions.txt"
matlab_rxns_file = "test/matlab_reactions.txt"

if !isfile(julia_rxns_file) || !isfile(matlab_rxns_file)
    println("ERROR: Reaction files not found!")
    println("Please run compare_julia_matlab_models.jl first.")
    exit(1)
end

julia_rxns = String.(readlines(julia_rxns_file))
matlab_rxns = String.(readlines(matlab_rxns_file))

julia_rxns = String.(filter(!isempty, strip.(julia_rxns)))
matlab_rxns = String.(filter(!isempty, strip.(matlab_rxns)))

println("Total reactions:")
println("  Julia:  $(length(julia_rxns))")
println("  MATLAB: $(length(matlab_rxns))")
println("  Difference: $(length(julia_rxns) - length(matlab_rxns))")
println()

# Function to extract base reaction name (without forward/backward suffixes only)
function get_base_reaction(rxn::String)
    # Only remove forward/backward suffixes to preserve reaction identity
    base = replace(rxn, r"_[fb]$" => "")
    return base
end

# Function to extract the original reaction name (before enzyme splitting)
function get_original_reaction(rxn::String)
    # Remove all enzyme and step suffixes
    base = replace(rxn, r"_e\d+" => "")
    base = replace(base, r"_s\d+" => "")
    base = replace(base, r"_p\d+" => "")
    base = replace(base, r"_s_p_transition" => "")
    base = replace(base, r"_[fb]$" => "")
    return base
end

# Count reactions per base reaction
function count_by_base(rxns::Vector{String})
    counts = Dict{String,Int}()
    for rxn in rxns
        base = get_base_reaction(rxn)
        counts[base] = get(counts, base, 0) + 1
    end
    return counts
end

julia_counts = count_by_base(julia_rxns)
matlab_counts = count_by_base(matlab_rxns)

# Also count by original reaction (fully stripped)
julia_original = count_by_base(map(get_original_reaction, julia_rxns))
matlab_original = count_by_base(map(get_original_reaction, matlab_rxns))

# Find reactions with different counts
all_bases = sort(unique([collect(keys(julia_counts)); collect(keys(matlab_counts))]))

differences = Tuple{String,Int,Int,Int}[]

for base in all_bases
    j_count = get(julia_counts, base, 0)
    m_count = get(matlab_counts, base, 0)
    diff = j_count - m_count

    if diff != 0
        push!(differences, (base, j_count, m_count, diff))
    end
end

# Sort by absolute difference
sort!(differences, by=x -> abs(x[4]), rev=true)

println("="^80)
println("REACTIONS WITH DIFFERENT COUNTS (sorted by difference)")
println("="^80)
println()
println("Base Reaction                    Julia  MATLAB  Diff")
println("-" * "="^79)

let total_diff = 0
    for (base, j_count, m_count, diff) in differences
        total_diff += diff
        sign = diff > 0 ? "+" : ""
        println("$(rpad(base, 32)) $(lpad(j_count, 5))  $(lpad(m_count, 6))  $(sign)$(lpad(diff, 4))")
    end

    println("-" * "="^79)
    println("TOTAL DIFFERENCE:                                      $(total_diff > 0 ? "+" : "")$(total_diff)")
end
println()

# Now analyze by original reaction (fully stripped)
println("="^80)
println("DIFFERENCES BY ORIGINAL REACTION (fully stripped)")
println("="^80)
println()

all_original = sort(unique([collect(keys(julia_original)); collect(keys(matlab_original))]))
original_diffs = Tuple{String,Int,Int,Int}[]

for base in all_original
    j_count = get(julia_original, base, 0)
    m_count = get(matlab_original, base, 0)
    diff = j_count - m_count

    if diff != 0
        push!(original_diffs, (base, j_count, m_count, diff))
    end
end

sort!(original_diffs, by=x -> abs(x[4]), rev=true)

println("Original Reaction            Julia  MATLAB  Diff")
println("-" * "="^79)

let total_diff = 0
    for (base, j_count, m_count, diff) in original_diffs[1:min(20, length(original_diffs))]
        total_diff += diff
        sign = diff > 0 ? "+" : ""
        println("$(rpad(base, 28)) $(lpad(j_count, 6))  $(lpad(m_count, 6))  $(sign)$(lpad(diff, 5))")
    end
    println("-" * "="^79)
    println("TOTAL (top 20):              $(lpad(sum(x[4] for x in original_diffs[1:min(20, length(original_diffs))]), 20))")
end
println()

# Focus on largest differences
println("="^80)
println("TOP 10 REACTIONS BY DIFFERENCE")
println("="^80)
println()

for (i, (base, j_count, m_count, diff)) in enumerate(differences[1:min(10, length(differences))])
    println("$i. $base")
    println("   Julia:  $j_count reactions")
    println("   MATLAB: $m_count reactions")
    println("   Difference: $(diff > 0 ? "+" : "")$diff")

    # Show example reactions
    julia_examples = filter(r -> startswith(r, base), julia_rxns)[1:min(3, j_count)]
    matlab_examples = filter(r -> startswith(r, base), matlab_rxns)[1:min(3, m_count)]

    if !isempty(julia_examples)
        println("   Julia examples:")
        for ex in julia_examples
            println("     - $ex")
        end
    end

    if !isempty(matlab_examples)
        println("   MATLAB examples:")
        for ex in matlab_examples
            println("     - $ex")
        end
    end
    println()
end

# Analyze by reaction type
println("="^80)
println("DIFFERENCES BY REACTION TYPE")
println("="^80)
println()

function count_by_type(rxns::Vector{String})
    catalytic = count(contains(r, "_s_p_transition") for r in rxns)
    substrate = count(r -> contains(r, "_s") && !contains(r, "_s_p_transition"), rxns)
    product = count(r -> contains(r, "_p") && !contains(r, "_s_p_transition"), rxns)
    forward = count(endswith(r, "_f") for r in rxns)
    backward = count(endswith(r, "_b") for r in rxns)
    unexpanded = length(rxns) - catalytic - substrate - product

    return (catalytic, substrate, product, forward, backward, unexpanded)
end

j_cat, j_sub, j_prod, j_fwd, j_bwd, j_unexp = count_by_type(julia_rxns)
m_cat, m_sub, m_prod, m_fwd, m_bwd, m_unexp = count_by_type(matlab_rxns)

println("Type                  Julia  MATLAB  Diff")
println("-" * "="^79)
println("$(rpad("Catalytic (_s_p_transition)", 20))  $(lpad(j_cat, 5))  $(lpad(m_cat, 6))  $(lpad(j_cat - m_cat > 0 ? "+$(j_cat - m_cat)" : "$(j_cat - m_cat)", 5))")
println("$(rpad("Substrate binding (_s)", 20))  $(lpad(j_sub, 5))  $(lpad(m_sub, 6))  $(lpad(j_sub - m_sub > 0 ? "+$(j_sub - m_sub)" : "$(j_sub - m_sub)", 5))")
println("$(rpad("Product release (_p)", 20))  $(lpad(j_prod, 5))  $(lpad(m_prod, 6))  $(lpad(j_prod - m_prod > 0 ? "+$(j_prod - m_prod)" : "$(j_prod - m_prod)", 5))")
println("$(rpad("Forward splits (_f)", 20))  $(lpad(j_fwd, 5))  $(lpad(m_fwd, 6))  $(lpad(j_fwd - m_fwd > 0 ? "+$(j_fwd - m_fwd)" : "$(j_fwd - m_fwd)", 5))")
println("$(rpad("Backward splits (_b)", 20))  $(lpad(j_bwd, 5))  $(lpad(m_bwd, 6))  $(lpad(j_bwd - m_bwd > 0 ? "+$(j_bwd - m_bwd)" : "$(j_bwd - m_bwd)", 5))")
println("$(rpad("Unexpanded", 20))  $(lpad(j_unexp, 5))  $(lpad(m_unexp, 6))  $(lpad(j_unexp - m_unexp > 0 ? "+$(j_unexp - m_unexp)" : "$(j_unexp - m_unexp)", 5))")
println()

# Analysis summary
println("="^80)
println("ANALYSIS SUMMARY")
println("="^80)
println()

cat_diff = j_cat - m_cat
elem_diff = (j_sub - m_sub) + (j_prod - m_prod)
split_diff = (j_fwd - m_fwd) # forward and backward should be same

println("Breakdown of 52-reaction difference:")
println("  1. Catalytic steps difference: $cat_diff")
println("  2. Elementary steps difference: $elem_diff")
println("  3. Forward/backward split difference: $(split_diff * 2) ($split_diff reactions × 2)")
println()

if cat_diff == 4
    println("✓ Catalytic step difference matches our bug fix expectation (4 extra)")
else
    println("⚠ Catalytic step difference is $cat_diff (expected 4)")
end

println()
println("The $(split_diff * 2)-reaction difference in forward/backward splits suggests:")
println("  - Julia split $split_diff more reactions than MATLAB")
println("  - This could be due to different blocking removal results")
println("  - OR different reversibility handling before irreversible conversion")
println()

println("="^80)
