"""
Test what note/annotation formats SBMLFBCModels accepts.
"""

using Pkg
Pkg.activate(".")

import AbstractFBCModels.CanonicalModel as CM
import SBMLFBCModels

println("="^80)
println("Testing SBML Export Format Requirements")
println("="^80)

# Test 1: Empty notes (baseline)
println("\nTest 1: No notes or annotations")
m1 = CM.Model()
m1.metabolites["test1"] = CM.Metabolite(name="Test1", compartment="c")
try
    sbml1 = convert(SBMLFBCModels.SBMLFBCModel, m1)
    println("✓ PASS - Empty notes/annotations work")
catch e
    println("✗ FAIL - $(typeof(e)): $(e.msg)")
end

# Test 2: Single-string note
println("\nTest 2: Single-string vector in notes")
m2 = CM.Model()
m2.metabolites["test2"] = CM.Metabolite(
    name="Test2",
    compartment="c",
    notes=Dict("description" => ["Single string entry"])
)
try
    sbml2 = convert(SBMLFBCModels.SBMLFBCModel, m2)
    println("✓ PASS - Single-string vector notes work")
catch e
    println("✗ FAIL - $(typeof(e)): $(e.msg)")
end

# Test 3: Multi-string vector in notes (suspected failure)
println("\nTest 3: Multi-string vector in notes")
m3 = CM.Model()
m3.metabolites["test3"] = CM.Metabolite(
    name="Test3",
    compartment="c",
    notes=Dict("components" => ["E100", "M_2pg_c"])
)
try
    sbml3 = convert(SBMLFBCModels.SBMLFBCModel, m3)
    println("✓ PASS - Multi-string vector notes work")
catch e
    println("✗ FAIL - $(typeof(e)): $(e.msg)")
end

# Test 4: Multi-string vector in annotations
println("\nTest 4: Multi-string vector in annotations")
m4 = CM.Model()
m4.metabolites["test4"] = CM.Metabolite(
    name="Test4",
    compartment="c",
    annotations=Dict("components" => ["E100", "M_2pg_c"])
)
try
    sbml4 = convert(SBMLFBCModels.SBMLFBCModel, m4)
    println("✓ PASS - Multi-string vector annotations work")
catch e
    println("✗ FAIL - $(typeof(e)): $(e.msg)")
end

# Test 5: Mixed - single string note + multi-string annotation
println("\nTest 5: Single-string note + multi-string annotation")
m5 = CM.Model()
m5.metabolites["test5"] = CM.Metabolite(
    name="Test5",
    compartment="c",
    notes=Dict("description" => ["Single string"]),
    annotations=Dict("components" => ["E100", "M_2pg_c"])
)
try
    sbml5 = convert(SBMLFBCModels.SBMLFBCModel, m5)
    println("✓ PASS - Single note + multi annotation work")
catch e
    println("✗ FAIL - $(typeof(e)): $(e.msg)")
end

# Test 6: Non-vector string in notes (plain string)
println("\nTest 6: Plain string (not vector) in notes")
m6 = CM.Model()
m6.metabolites["test6"] = CM.Metabolite(
    name="Test6",
    compartment="c",
    notes=Dict("description" => "Plain string not in vector")
)
try
    sbml6 = convert(SBMLFBCModels.SBMLFBCModel, m6)
    println("✓ PASS - Plain string notes work")
catch e
    println("✗ FAIL - $(typeof(e)): $(e.msg)")
end

println("\n" * "="^80)
println("Summary:")
println("- If Test 3 fails but Test 4 passes → move multi-value data to annotations")
println("- If both Test 3 and 4 fail → must use single-string format")
println("="^80)
