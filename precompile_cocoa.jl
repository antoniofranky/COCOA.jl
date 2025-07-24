#!/usr/bin/env julia
"""
COCOA.jl Precompilation Script for HPC
=====================================

This script precompiles COCOA.jl and its dependencies to reduce startup overhead 
on HPC systems. Run this once before your main computation jobs.

Usage:
    julia --project=/work/schaffran1/COCOA.jl precompile_cocoa.jl
"""

using Pkg

println("COCOA.jl HPC Precompilation Script")
println("=" ^ 40)

# Activate the COCOA environment
println("Activating COCOA.jl environment...")
Pkg.activate("/work/schaffran1/COCOA.jl")

# Update and instantiate dependencies
println("Updating and instantiating dependencies...")
Pkg.instantiate()
Pkg.update()

# Precompile all dependencies
println("Precompiling all dependencies...")
Pkg.precompile()

# Load and compile main modules
println("Loading and compiling main modules...")
using COCOA
using COBREXA
using HiGHS
using SBMLFBCModels
using AbstractFBCModels
using JLD2
using DataFrames
using Distributed

println("Loading model for compilation...")
# Load a small model to trigger compilation of model-specific functions
test_model_path = "/work/schaffran1/COCOA.jl/test/e_coli_core.xml"
if isfile(test_model_path)
    model = COBREXA.load_model(test_model_path)
    println("✓ Model loaded successfully")
    
    # Trigger compilation of key functions with small parameters
    println("Compiling concordance analysis functions...")
    try
        results = concordance_analysis(
            model;
            optimizer=HiGHS.Optimizer,
            sample_size=5,          # Minimal for compilation
            stage_size=10,
            batch_size=5,
            tolerance=0.01,
            seed=42
        )
        println("✓ Concordance analysis compiled successfully")
    catch e
        println("⚠ Concordance analysis compilation had issues: $e")
        println("  This is normal for the compilation run")
    end
else
    println("⚠ Test model not found at $test_model_path")
    println("  Skipping model-specific compilation")
end

# Force compilation of distributed computing functions
println("Compiling distributed computing functions...")
if nworkers() > 1
    @everywhere begin
        using COCOA
        using HiGHS
    end
    println("✓ Distributed functions compiled")
else
    println("⚠ No workers available for distributed compilation")
end

# Create system image (optional, commented out as it requires PackageCompiler.jl)
# println("Creating system image (optional)...")
# using PackageCompiler
# create_sysimage(
#     ["COCOA", "COBREXA", "HiGHS", "SBMLFBCModels", "AbstractFBCModels"];
#     sysimage_path="/work/schaffran1/cocoa_sysimage.so",
#     project="/work/schaffran1/COCOA.jl"
# )

println("\n" * "=" ^ 40)
println("✅ COCOA.jl precompilation complete!")
println("=" ^ 40)

println("\nTo use the precompiled environment:")
println("  julia --project=/work/schaffran1/COCOA.jl your_script.jl")
println("\nFor maximum performance, also use these flags:")
println("  --startup-file=no --history-file=no --compiled-modules=yes --optimize=2")

# Clean up compilation artifacts if needed
println("\nCleaning up compilation artifacts...")
GC.gc()

println("Precompilation script completed successfully! 🚀")