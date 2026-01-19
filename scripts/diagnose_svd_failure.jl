"""
Diagnostic script to investigate SVD failures in kinetic analysis.

Run this on a model that fails with LAPACKException to understand the root cause.

Usage:
    julia --project=. scripts/diagnose_svd_failure.jl <result_file.jld2> <model_dir>
"""

using Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using COCOA, COBREXA, JLD2
using SBMLFBCModels, AbstractFBCModels
using LinearAlgebra, SparseArrays

function diagnose_matrix(M::AbstractMatrix, name::String)
    println("\n" * "="^60)
    println("Matrix: $name")
    println("="^60)

    println("Size: $(size(M))")
    println("Element type: $(eltype(M))")

    if M isa SparseArrays.AbstractSparseMatrix
        println("Sparse with $(nnz(M)) nonzeros ($(round(100*nnz(M)/length(M), digits=2))% fill)")
        nz = nonzeros(M)
    else
        println("Dense matrix")
        nz = vec(M)
    end

    # Value statistics
    finite_vals = filter(isfinite, nz)
    println("\nValue statistics:")
    println("  Any NaN: $(any(isnan, nz))")
    println("  Any Inf: $(any(isinf, nz))")
    println("  Any zero: $(any(iszero, nz))")

    if !isempty(finite_vals)
        nonzero_vals = filter(!iszero, finite_vals)
        if !isempty(nonzero_vals)
            println("  Min nonzero: $(minimum(abs.(nonzero_vals)))")
            println("  Max: $(maximum(abs.(nonzero_vals)))")
            println("  Value range: $(maximum(abs.(nonzero_vals)) / minimum(abs.(nonzero_vals)))")
        end
    end

    # Try different SVD algorithms
    println("\nSVD attempts:")
    M_dense = Matrix{Float64}(M)

    # Default (divide and conquer)
    print("  DivideAndConquer: ")
    try
        svd_result = svd(M_dense)
        println("SUCCESS - rank=$(count(s -> s > 1e-10 * maximum(svd_result.S), svd_result.S))")
        println("    Singular values (first 10): $(round.(svd_result.S[1:min(10, length(svd_result.S))], sigdigits=4))")
        println("    Condition number: $(svd_result.S[1] / svd_result.S[end])")
    catch e
        err_info = hasproperty(e, :info) ? e.info : string(e)
        println("FAILED - $(typeof(e)): $err_info")
    end

    # QR iteration
    print("  QRIteration: ")
    try
        svd_result = svd(M_dense; alg=QRIteration())
        println("SUCCESS - rank=$(count(s -> s > 1e-10 * maximum(svd_result.S), svd_result.S))")
    catch e
        println("FAILED - $(typeof(e))")
    end

    # QR factorization as alternative
    print("  QR rank estimate: ")
    try
        qr_result = qr(M_dense, ColumnNorm())
        R_diag = abs.(diag(qr_result.R))
        max_val = maximum(R_diag)
        rank_est = count(r -> r > 1e-10 * max_val, R_diag)
        println("SUCCESS - rank≈$rank_est")
    catch e
        println("FAILED - $(typeof(e))")
    end
end

function main()
    if length(ARGS) < 2
        error("Usage: julia diagnose_svd_failure.jl <result_file.jld2> <model_dir>")
    end

    result_file = ARGS[1]
    model_dir = ARGS[2]

    println("Loading concordance results from: $result_file")
    data = JLD2.load(result_file)
    results = data["results"]

    # Extract model name
    model_name = get(data, "model_name", nothing)
    if model_name === nothing
        basename_str = basename(result_file)
        rest = basename_str[17:end]
        m = match(r"^(.+)_\d+_\d+_cv", rest)
        model_name = m !== nothing ? m.captures[1] : error("Could not extract model name")
    end

    println("Model name: $model_name")

    # Find model file
    possible_paths = [
        joinpath(model_dir, "$(model_name).xml"),
        joinpath(model_dir, "random_0", "$(model_name).xml"),
    ]
    model_file = nothing
    for path in possible_paths
        if isfile(path)
            model_file = path
            break
        end
    end
    model_file === nothing && error("Model file not found")

    println("Loading model from: $model_file")
    model = COBREXA.load_model(model_file)

    # Extract concordance modules
    println("\nExtracting concordance modules...")
    concordance_modules = COCOA.extract_concordance_modules(results)
    println("Balanced complexes: $(length(concordance_modules[1]))")
    println("Unbalanced modules: $(length(concordance_modules) - 1)")

    # Build the network (same as kinetic_analysis does internally)
    println("\nBuilding network matrices...")

    # Get the incidence and stoichiometry matrices using COCOA's functions
    A_matrix = COCOA.incidence(model)
    Y_matrix = COCOA.complex_stoichiometry(model)

    println("Incidence matrix A: $(size(A_matrix))")
    println("Complex stoichiometry Y: $(size(Y_matrix))")

    diagnose_matrix(A_matrix, "Incidence matrix A")
    diagnose_matrix(Y_matrix, "Complex stoichiometry Y")

    # Compute S = Y * A
    S_matrix = Y_matrix * A_matrix
    diagnose_matrix(S_matrix, "Stoichiometry S = Y*A")

    println("\n" * "="^60)
    println("DIAGNOSIS COMPLETE")
    println("="^60)
    println("\nIf DivideAndConquer fails but QRIteration succeeds, the fix is")
    println("to use QRIteration as fallback in robust_rank().")
    println("\nIf both fail, there may be numerical issues with the model or")
    println("the matrix construction that need investigation.")
end

main()
